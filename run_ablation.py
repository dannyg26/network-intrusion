"""
run_ablation.py
Runs all 5 ablation configurations sequentially and prints a comparison table.

Ablation plan (from proposal):
  1. MLP baseline                   (plain cross-entropy, class weights)
  2. MLP + SMOTE
  3. MLP + Focal Loss
  4. CNN-LSTM (no extra tricks)
  5. CNN-LSTM + SMOTE + Focal Loss + cosine LR schedule  (full model)

Usage:
    python run_ablation.py [--epochs 20] [--batch-size 256]
"""

import argparse
import subprocess
import sys

from evaluate import load_results, print_table, plot_curves


EXPERIMENTS = [
    # (name, model, smote, focal, lr_schedule, focal_gamma, lr_override)
    # lr_override=None means use the global --lr arg; otherwise that LR is used for this run.

    # ── Original ablation (kept for comparison) ────────────────────────────────
    ("baseline",             "mlp",           False, False, "none",    1.0, None),
    ("baseline_smote",       "mlp",           True,  False, "none",    1.0, None),
    ("baseline_focal",       "mlp",           False, True,  "none",    1.0, None),
    ("cnn_lstm",             "cnn_lstm",      False, False, "none",    1.0, None),
    ("cnn_lstm_full",        "cnn_lstm",      True,  True,  "cosine",  1.0, None),

    # ── Improved experiments ───────────────────────────────────────────────────
    # ResMLPModel: wider + residual connections, no tricks
    ("res_mlp",              "res_mlp",       False, False, "none",    1.0, None),

    # CNN-LSTM with self-attention, no tricks (ceiling test)
    ("cnn_lstm_attn",        "cnn_lstm_attn", False, False, "none",    1.0, None),

    # Best candidate: attention + cosine LR at 1e-4 (finer convergence past plateau)
    ("cnn_lstm_attn_cosine", "cnn_lstm_attn", False, False, "cosine",  1.0, 1e-4),

    # Full improved model: attention + balanced SMOTE + cosine (no focal loss)
    ("cnn_lstm_attn_smote",  "cnn_lstm_attn", True,  False, "cosine",  1.0, 1e-4),
]


def run_experiment(name, model, smote, focal, lr_schedule, focal_gamma, lr_override,
                   epochs, batch_size, lr, patience, data_dir, results_dir):
    effective_lr = lr_override if lr_override is not None else lr
    cmd = [
        sys.executable, "train.py",
        "--experiment",  name,
        "--model",       model,
        "--lr-schedule", lr_schedule,
        "--epochs",      str(epochs),
        "--batch-size",  str(batch_size),
        "--lr",          str(effective_lr),
        "--patience",    str(patience),
        "--data-dir",    data_dir,
        "--results-dir", results_dir,
    ]
    if smote:
        cmd.append("--smote")
    if focal:
        cmd.extend(["--focal-loss", "--focal-gamma", str(focal_gamma)])

    print(f"\n{'#'*60}")
    print(f"# Starting: {name}  (lr={effective_lr})")
    print(f"# Command : {' '.join(cmd)}")
    print(f"{'#'*60}")
    result = subprocess.run(cmd)
    return result.returncode == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch-size",  type=int,   default=256)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--patience",    type=int,   default=10)
    parser.add_argument("--data-dir",    default="processed")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--skip-done",   action="store_true",
                        help="Skip experiments that already have results.json")
    args = parser.parse_args()

    import os
    successes, failures = [], []

    for name, model, smote, focal, lr_sched, focal_gamma, lr_override in EXPERIMENTS:
        result_file = os.path.join(args.results_dir, name, "results.json")
        if args.skip_done and os.path.exists(result_file):
            print(f"Skipping '{name}' (already done).")
            successes.append(name)
            continue

        ok = run_experiment(
            name, model, smote, focal, lr_sched, focal_gamma, lr_override,
            args.epochs, args.batch_size, args.lr, args.patience,
            args.data_dir, args.results_dir,
        )
        (successes if ok else failures).append(name)

    print(f"\n\nAll experiments finished. "
          f"Success: {len(successes)}  |  Failed: {len(failures)}")
    if failures:
        print(f"Failed experiments: {failures}")

    # ── Print final comparison table ─────────────────────────────────────────
    records = load_results(args.results_dir)
    # Sort by the ablation order
    order = {name: i for i, (name, *_) in enumerate(EXPERIMENTS)}
    records.sort(key=lambda r: order.get(r["experiment"], 99))
    print_table(records)

    # ── Plot training curves ──────────────────────────────────────────────────
    plot_curves(args.results_dir, records)
