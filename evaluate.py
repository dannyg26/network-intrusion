"""
evaluate.py
Loads saved experiment results and prints a clean ablation comparison table.
Also generates a confusion matrix and training curve plots (if matplotlib available).

Usage:
    python evaluate.py                         # scans results/ directory
    python evaluate.py --results-dir results
"""

import argparse
import glob
import json
import os
import pickle

import numpy as np


def load_results(results_dir: str) -> list[dict]:
    pattern = os.path.join(results_dir, "**", "results.json")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        print(f"No results.json files found under '{results_dir}'.")
        return []
    records = []
    for f in files:
        with open(f) as fp:
            records.append(json.load(fp))
    return records


def print_table(records: list[dict]):
    if not records:
        return

    header = (
        f"{'Experiment':<25} {'Model':<10} {'SMOTE':<6} {'Focal':<6} "
        f"{'LR Sched':<10} {'Acc':>7} {'F1-Mac':>7} {'F1-Wgt':>7} {'AUC':>7}"
    )
    sep = "-" * len(header)
    print(f"\n{'ABLATION STUDY — TEST SET RESULTS':^{len(header)}}")
    print(sep)
    print(header)
    print(sep)

    for r in records:
        print(
            f"{r['experiment']:<25} "
            f"{r['model']:<10} "
            f"{'Y' if r['smote']      else 'N':<6} "
            f"{'Y' if r['focal_loss'] else 'N':<6} "
            f"{r['lr_schedule']:<10} "
            f"{r.get('test_accuracy', float('nan')):>7.4f} "
            f"{r.get('f1_macro',      float('nan')):>7.4f} "
            f"{r.get('f1_weighted',   float('nan')):>7.4f} "
            f"{r.get('roc_auc',       float('nan')):>7.4f}"
        )
    print(sep)


def plot_curves(results_dir: str, records: list[dict]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10.colors

    for i, r in enumerate(records):
        history_path = os.path.join(results_dir, r["experiment"], "history.json")
        if not os.path.exists(history_path):
            continue
        with open(history_path) as f:
            h = json.load(f)
        label = r["experiment"]
        color = colors[i % len(colors)]
        epochs = range(1, len(h["train_acc"]) + 1)
        axes[0].plot(epochs, h["val_acc"],  label=label, color=color)
        axes[1].plot(epochs, h["val_loss"], label=label, color=color)

    axes[0].set_title("Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=7)
    axes[0].grid(True)

    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=7)
    axes[1].grid(True)

    out = os.path.join(results_dir, "training_curves.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Training curves saved → {out}")


def plot_confusion(results_dir: str, experiment: str, data_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from models import build_model
    except ImportError as e:
        print(f"Cannot plot confusion matrix: {e}")
        return

    result_path = os.path.join(results_dir, experiment, "results.json")
    model_path  = os.path.join(results_dir, experiment, "best_model.pt")
    if not os.path.exists(result_path) or not os.path.exists(model_path):
        return

    with open(result_path) as f:
        r = json.load(f)
    with open(os.path.join(data_dir, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    splits = np.load(os.path.join(data_dir, "splits.npz"))
    X_test, y_test = splits["X_test"], splits["y_test"]
    num_classes = len(le.classes_)

    model = build_model(r["model"], X_test.shape[1], num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).long())
    loader = DataLoader(ds, batch_size=512, shuffle=False)

    preds, targets = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            preds.append(model(X_b).argmax(1).numpy())
            targets.append(y_b.numpy())
    preds   = np.concatenate(preds)
    targets = np.concatenate(targets)

    cm = confusion_matrix(targets, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {experiment}")
    plt.tight_layout()
    out = os.path.join(results_dir, experiment, "confusion_matrix.png")
    plt.savefig(out, dpi=150)
    print(f"Confusion matrix saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--data-dir",    default="processed",
                        help="Needed for confusion matrix plots")
    parser.add_argument("--plot",        action="store_true",
                        help="Generate training curve plots")
    parser.add_argument("--confusion",   default=None,
                        help="Experiment name to plot confusion matrix for")
    args = parser.parse_args()

    records = load_results(args.results_dir)
    print_table(records)

    if args.plot:
        plot_curves(args.results_dir, records)

    if args.confusion:
        plot_confusion(args.results_dir, args.confusion, args.data_dir)
