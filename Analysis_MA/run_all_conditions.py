"""
run_all_conditions.py
=====================
Trains and evaluates a DLinear model for each cohort/condition combination,
mirroring the pipeline in run_linear_v1.py but looping over:

  SFC_S  (Sham)         SFC_F  (Formalin)
  HNG_H  (Hargreaves)   HNG_N  (Neuropathic)

Expected CSV files in Analysis_MA/data/:
  SFC_S_normalized.csv, SFC_F_normalized.csv,
  HNG_H_normalized.csv, HNG_N_normalized.csv

Each CSV has columns: date, F1, F2, ..., F66

Usage:
  python Analysis_MA/run_all_conditions.py
"""

import os
import sys
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

# Ensure the repo root is on the path so we can import exp, utils, etc.
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

from utils.tools import dotdict
from exp.exp_main import Exp_Main

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# ---------------------------------------------------------------------------
# Create log directories
# ---------------------------------------------------------------------------
log_dir = os.path.join(repo_root, "logs", "LongForecasting")
os.makedirs(log_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Define the conditions to run
# ---------------------------------------------------------------------------
conditions = [
    {"cohort": "SFC", "condition": "S", "label": "Sham"},
    {"cohort": "SFC", "condition": "F", "label": "Formalin"},
    {"cohort": "HNG", "condition": "H", "label": "Hargreaves"},
    {"cohort": "HNG", "condition": "N", "label": "Neuropathic"},
]

# ---------------------------------------------------------------------------
# Model / training hyperparameters (same as run_linear_v1.py)
# ---------------------------------------------------------------------------
seq_len = 96          # Input window size (~7.7 seconds at 12.5 Hz)
pred_len = 12         # Prediction horizon (1 second)
n_features = 66       # Number of features
model_name = "DLinear"

# ---------------------------------------------------------------------------
# Results collection
# ---------------------------------------------------------------------------
all_results = {}

for cond_info in conditions:
    cohort = cond_info["cohort"]
    condition = cond_info["condition"]
    label = cond_info["label"]
    tag = f"{cohort}_{condition}"

    csv_file = f"{tag}_normalized.csv"
    csv_path = os.path.join(repo_root, "Analysis_MA", "data", csv_file)

    if not os.path.isfile(csv_path):
        print(f"[SKIP] CSV not found for {tag}: {csv_path}")
        continue

    print(f"\n{'='*70}")
    print(f"  Running: {tag} ({label})")
    print(f"  Data:    {csv_path}")
    print(f"{'='*70}\n")

    # --- Build args (same structure as run_linear_v1.py) -----------------
    args = dotdict()

    # Basic configuration
    args.is_training = 1
    args.model_id = f"MA_{tag}_{seq_len}_{pred_len}"
    args.model = model_name
    args.train_only = False

    # Data loader configuration
    args.data = "custom"
    args.root_path = os.path.join(repo_root, "Analysis_MA", "data")
    args.data_path = csv_file
    args.features = "M"
    args.target = "F66"
    args.freq = "ms"
    args.checkpoints = os.path.join(repo_root, "Analysis_MA", "checkpoints")

    # Forecasting task
    args.seq_len = seq_len
    args.label_len = 48   # Not used for linear models
    args.pred_len = pred_len

    # Model parameters
    args.individual = True
    args.enc_in = n_features
    args.dec_in = n_features
    args.c_out = n_features
    args.moving_avg = 5

    # Training parameters
    args.num_workers = 0
    args.itr = 1
    args.train_epochs = 10
    args.batch_size = 32
    args.patience = 3
    args.learning_rate = 0.0001
    args.loss = "mse"
    args.lradj = "type1"
    args.use_amp = False
    args.des = "Exp"

    # GPU configuration
    args.use_gpu = True if torch.cuda.is_available() else False
    args.gpu = 0
    args.use_multi_gpu = False

    # --- Train & Test ----------------------------------------------------
    setting = "{}_{}_{}_ft{}_sl{}_pl{}_in{}_it{}_lr{}_bs{}".format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        args.individual,
        args.itr,
        args.learning_rate,
        args.batch_size,
    )

    log_file = os.path.join(log_dir, f"{model_name}_I_{tag}_{seq_len}_{pred_len}.log")

    print(f"Args: {args}\n")
    print(f"Setting: {setting}")
    print(f"Log file: {log_file}\n")

    exp = Exp_Main(args)

    print(f">>> Start training: {setting}")
    exp.train(setting)

    if not args.train_only:
        print(f">>> Testing: {setting}")
        exp.test(setting)

    torch.cuda.empty_cache()

    # --- Predict & save --------------------------------------------------
    exp.predict(setting, True)

    results_dir = os.path.join(repo_root, "results", setting)
    preds = np.load(os.path.join(results_dir, "pred.npy"))
    trues = np.load(os.path.join(results_dir, "true.npy"))

    all_results[tag] = {
        "setting": setting,
        "preds": preds,
        "trues": trues,
        "label": label,
    }

    # Quick sanity-check plot (feature 5)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(trues[0, :, 5], label="GroundTruth")
    ax.plot(preds[0, :, 5], label="Prediction")
    ax.set_title(f"{tag} ({label}) - Feature 6, sample 0")
    ax.legend()
    fig_path = os.path.join(repo_root, "Analysis_MA", f"sanity_{tag}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Sanity plot saved: {fig_path}")

    # Write to log
    with open(log_file, "w") as f:
        f.write(f"Condition: {tag} ({label})\n")
        f.write(f"Setting:   {setting}\n")
        f.write(f"Pred shape: {preds.shape}\n")
        f.write(f"True shape: {trues.shape}\n")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("  SUMMARY")
print(f"{'='*70}")
for tag, res in all_results.items():
    preds = res["preds"]
    trues = res["trues"]
    mse = np.mean((preds - trues) ** 2)
    mae = np.mean(np.abs(preds - trues))
    print(f"  {tag:8s} ({res['label']:12s}) | MSE={mse:.6f}  MAE={mae:.6f}  shape={preds.shape}")

print(f"\nAll results saved in: {os.path.join(repo_root, 'results')}")
print("Done.")
