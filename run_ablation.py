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
    # (name,                model,           smote,  focal, lr_schedule, focal_gamma)
    ("baseline",            "mlp",           False, False, "none",    1.0),
    ("baseline_smote",      "mlp",           True,  False, "none",    1.0),
    ("baseline_focal",      "mlp",           False, True,  "none",    1.0),
    ("cnn_lstm",            "cnn_lstm",      False, False, "none",    1.0),
    ("cnn_lstm_full",       "cnn_lstm",      True,  True,  "cosine",  1.0),
    # New: improved models
    ("res_mlp",             "res_mlp",       False, False, "none",    1.0),
    ("res_mlp_focal",       "res_mlp",       False, True,  "cosine",  1.0),
    ("cnn_lstm_attn",       "cnn_lstm_attn", False, False, "none",    1.0),
    ("cnn_lstm_attn_full",  "cnn_lstm_attn", False, True,  "cosine",  1.0),
]


def run_experiment(name, model, smote, focal, lr_schedule, focal_gamma, epochs,
                   batch_size, lr, data_dir, results_dir):
    cmd = [
        sys.executable, "train.py",
        "--experiment",  name,
        "--model",       model,
        "--lr-schedule", lr_schedule,
        "--epochs",      str(epochs),
        "--batch-size",  str(batch_size),
        "--lr",          str(lr),
        "--data-dir",    data_dir,
        "--results-dir", results_dir,
    ]
    if smote:
        cmd.append("--smote")
    if focal:
        cmd.extend(["--focal-loss", "--focal-gamma", str(focal_gamma)])

    print(f"\n{'#'*60}")
    print(f"# Starting: {name}")
    print(f"# Command : {' '.join(cmd)}")
    print(f"{'#'*60}")
    result = subprocess.run(cmd)
    return result.returncode == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch-size",  type=int,   default=256)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--data-dir",    default="processed")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--skip-done",   action="store_true",
                        help="Skip experiments that already have results.json")
    args = parser.parse_args()

    import os
    successes, failures = [], []

    for name, model, smote, focal, lr_sched, focal_gamma in EXPERIMENTS:
        result_file = os.path.join(args.results_dir, name, "results.json")
        if args.skip_done and os.path.exists(result_file):
            print(f"Skipping '{name}' (already done).")
            successes.append(name)
            continue

        ok = run_experiment(
            name, model, smote, focal, lr_sched, focal_gamma,
            args.epochs, args.batch_size, args.lr,
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
