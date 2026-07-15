#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
P="${P:-31}"
TRAIN_FRAC="${TRAIN_FRAC:-0.4}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-3e-4}"
BATCH_SIZE="${BATCH_SIZE:-128}"
D_MODEL="${D_MODEL:-64}"
N_LAYERS="${N_LAYERS:-2}"
N_HEADS="${N_HEADS:-4}"
SEED="${SEED:-42}"

"$PYTHON_BIN" modp_verifier_experiment.py \
  --p "$P" \
  --train_frac "$TRAIN_FRAC" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --batch_size "$BATCH_SIZE" \
  --d_model "$D_MODEL" \
  --n_layers "$N_LAYERS" \
  --n_heads "$N_HEADS" \
  --seed "$SEED" \
  --out_dir outputs \
  --checkpoint_dir checkpoints \
  --alpha_beta 1.0,0.0 0.95,0.05 0.9,0.1 0.8,0.2 0.6,0.4
