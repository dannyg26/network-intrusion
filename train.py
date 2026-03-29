"""
train.py
Training script for the network intrusion detection project.

Examples
--------
# MLP baseline (plain cross-entropy, no SMOTE)
python train.py --model mlp --experiment baseline

# MLP + SMOTE
python train.py --model mlp --smote --experiment baseline_smote

# MLP + Focal Loss
python train.py --model mlp --focal-loss --experiment baseline_focal

# CNN-LSTM (full model: SMOTE + Focal Loss + LR schedule)
python train.py --model cnn_lstm --smote --focal-loss --lr-schedule cosine --experiment cnn_lstm_full
"""

import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models import build_model

# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Multi-class focal loss. Focuses on hard, misclassified examples."""

    def __init__(self, gamma: float = 2.0, weight=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(
            logits, targets, weight=self.weight, reduction="none"
        )
        pt = torch.exp(-ce)                      # probability of correct class
        focal = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return focal.mean()
        return focal.sum()


# ── SMOTE (via imbalanced-learn) ──────────────────────────────────────────────
def apply_smote(X_train: np.ndarray, y_train: np.ndarray, cap: int = 50_000):
    """
    Apply SMOTE with a per-class cap so minority classes are oversampled up to
    `cap` samples — avoiding the O(n^2) cost of balancing against a ~480k DDoS class.
    """
    from imblearn.over_sampling import SMOTE
    counts = np.bincount(y_train)
    target = min(int(counts.max()), cap)
    strategy = {cls: max(target, cnt) for cls, cnt in enumerate(counts)}
    print(f"  Applying SMOTE (target per class: {target:,})...", flush=True)
    t0 = time.time()
    sm = SMOTE(random_state=42, sampling_strategy=strategy)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"  SMOTE done ({time.time()-t0:.1f}s): {len(X_train):,} -> {len(X_res):,} samples")
    return X_res.astype(np.float32), y_res


# ── Helpers ───────────────────────────────────────────────────────────────────
def make_loader(X, y, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


def class_weights(y_train: np.ndarray, num_classes: int, device) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_targets = [], []
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        logits = model(X_b)
        loss = criterion(logits, y_b)
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * len(y_b)
        correct += (preds == y_b).sum().item()
        n += len(y_b)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(y_b.cpu().numpy())
    return (
        total_loss / n,
        correct / n,
        np.concatenate(all_preds),
        np.concatenate(all_targets),
    )


# ── Main training loop ────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Experiment : {args.experiment}")
    print(f"Model      : {args.model}")
    print(f"SMOTE      : {args.smote}")
    print(f"Focal Loss : {args.focal_loss}")
    print(f"LR Schedule: {args.lr_schedule}")
    print(f"Device     : {device}")
    print(f"{'='*60}\n")

    # ── Load preprocessed data ────────────────────────────────────────────────
    splits_path = os.path.join(args.data_dir, "splits.npz")
    if not os.path.exists(splits_path):
        raise FileNotFoundError(
            f"'{splits_path}' not found. Run preprocess.py first."
        )
    splits = np.load(splits_path)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val,   y_val   = splits["X_val"],   splits["y_val"]
    X_test,  y_test  = splits["X_test"],  splits["y_test"]

    with open(os.path.join(args.data_dir, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    num_classes = len(le.classes_)
    input_dim   = X_train.shape[1]
    print(f"Input dim: {input_dim}  |  Classes: {num_classes}  |  "
          f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}\n")

    # ── SMOTE ─────────────────────────────────────────────────────────────────
    if args.smote:
        X_train, y_train = apply_smote(X_train, y_train)

    # ── Dataloaders ───────────────────────────────────────────────────────────
    train_loader = make_loader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   args.batch_size, shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  args.batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args.model, input_dim, num_classes, dropout=0.3)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    cw = class_weights(y_train, num_classes, device)
    if args.focal_loss:
        criterion = FocalLoss(gamma=args.focal_gamma, weight=cw)
        print(f"Loss: Focal Loss (gamma={args.focal_gamma}, class-weighted)")
    else:
        criterion = nn.CrossEntropyLoss(weight=cw)
        print("Loss: Weighted Cross-Entropy")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ── LR Scheduler ─────────────────────────────────────────────────────────
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
        print(f"LR Schedule: Cosine Annealing (T_max={args.epochs})")
    elif args.lr_schedule == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5, verbose=True
        )
        print("LR Schedule: ReduceLROnPlateau (patience=3, factor=0.5)")
    else:
        scheduler = None
        print("LR Schedule: None (fixed)")

    # ── Output dir ───────────────────────────────────────────────────────────
    exp_dir = os.path.join(args.results_dir, args.experiment)
    os.makedirs(exp_dir, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc    = 0.0
    best_epoch      = 0
    patience_counter = 0

    print(f"\n{'Ep':>4}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>9}  {'Val Acc':>8}  {'LR':>10}  {'Time':>7}")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        ep_loss, ep_correct, ep_n = 0.0, 0, 0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ep_loss    += loss.item() * len(y_b)
            ep_correct += (logits.argmax(1) == y_b).sum().item()
            ep_n       += len(y_b)

        train_loss = ep_loss / ep_n
        train_acc  = ep_correct / ep_n
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        # Scheduler step
        if scheduler is not None:
            if args.lr_schedule == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"{epoch:>4d}  {train_loss:>10.4f}  {train_acc:>9.4f}  "
              f"{val_loss:>9.4f}  {val_acc:>8.4f}  {current_lr:>10.2e}  {elapsed:>6.1f}s")

        # Save best model + early stopping
        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            best_epoch       = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pt"))
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"\nEarly stopping triggered (no improvement for {args.patience} epochs).")
                break

    print(f"\nBest val acc: {best_val_acc:.4f} at epoch {best_epoch}")

    # ── Test evaluation with best model ───────────────────────────────────────
    model.load_state_dict(torch.load(os.path.join(exp_dir, "best_model.pt"),
                                     map_location=device))
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, roc_auc_score
    )
    from sklearn.preprocessing import label_binarize

    test_loss, test_acc, preds, targets = evaluate(model, test_loader, criterion, device)

    prec_macro  = precision_score(targets, preds, average="macro",  zero_division=0)
    rec_macro   = recall_score   (targets, preds, average="macro",  zero_division=0)
    f1_macro    = f1_score       (targets, preds, average="macro",  zero_division=0)
    f1_weighted = f1_score       (targets, preds, average="weighted", zero_division=0)

    # ROC-AUC (one-vs-rest, macro)
    y_bin = label_binarize(targets, classes=list(range(num_classes)))
    # need probabilities - re-run inference
    model.eval()
    all_probs = []
    with torch.no_grad():
        for X_b, _ in test_loader:
            logits = model(X_b.to(device))
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    probs = np.concatenate(all_probs, axis=0)
    try:
        roc_auc = roc_auc_score(y_bin, probs, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = float("nan")

    print(f"\n{'-'*50}")
    print(f"TEST RESULTS -- {args.experiment}")
    print(f"{'-'*50}")
    print(f"  Accuracy         : {test_acc:.4f}")
    print(f"  Precision (macro): {prec_macro:.4f}")
    print(f"  Recall (macro)   : {rec_macro:.4f}")
    print(f"  F1 (macro)       : {f1_macro:.4f}")
    print(f"  F1 (weighted)    : {f1_weighted:.4f}")
    print(f"  ROC-AUC (macro)  : {roc_auc:.4f}")
    print(f"\nPer-class report:")
    print(classification_report(targets, preds, target_names=le.classes_, zero_division=0))

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "experiment":    args.experiment,
        "model":         args.model,
        "smote":         args.smote,
        "focal_loss":    args.focal_loss,
        "focal_gamma":   args.focal_gamma,
        "lr_schedule":   args.lr_schedule,
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "lr":            args.lr,
        "best_val_acc":  best_val_acc,
        "best_epoch":    best_epoch,
        "test_accuracy": test_acc,
        "precision_macro": prec_macro,
        "recall_macro":    rec_macro,
        "f1_macro":        f1_macro,
        "f1_weighted":     f1_weighted,
        "roc_auc":         roc_auc,
    }
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(exp_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nResults saved to '{exp_dir}/'")
    return results


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train intrusion detection model")
    parser.add_argument("--data-dir",    default="processed",
                        help="Directory containing splits.npz (output of preprocess.py)")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--experiment",  default="experiment",
                        help="Name for this run (used as subdirectory in results/)")
    parser.add_argument("--model",       default="mlp",
                        choices=["mlp", "res_mlp", "cnn_lstm", "cnn_lstm_attn"])
    parser.add_argument("--smote",       action="store_true")
    parser.add_argument("--focal-loss",  action="store_true")
    parser.add_argument("--focal-gamma", type=float, default=1.0,
                        help="Focal loss gamma (default 1.0; 2.0 is too aggressive)")
    parser.add_argument("--lr-schedule", default="none",
                        choices=["none", "cosine", "plateau"])
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch-size",  type=int,   default=256)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--patience",    type=int,   default=7,
                        help="Early stopping patience (0 to disable)")
    args = parser.parse_args()
    train(args)
