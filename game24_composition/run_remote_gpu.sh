#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

: "${GPU_HOST:?Set GPU_HOST in .env}"

GPU_SSH_USER="${GPU_SSH_USER:-ubuntu}"
REMOTE_DIR="${REMOTE_DIR:-~/game24_composition}"
REMOTE="${GPU_SSH_USER}@${GPU_HOST}"

ssh "$REMOTE" "mkdir -p $REMOTE_DIR"
rsync -az --delete \
  --exclude '.env' \
  --exclude '.venv' \
  --exclude 'cloud_artifacts' \
  --exclude 'runs/*' \
  --exclude 'outputs/*' \
  --exclude 'data/synthetic/*' \
  ./ "$REMOTE:$REMOTE_DIR/"

ssh "$REMOTE" "cd $REMOTE_DIR && python3 -m venv .venv && . .venv/bin/activate && pip install -e . && bash run_first5.sh 2>&1 | tee run_first5.log"

mkdir -p cloud_artifacts
rsync -az "$REMOTE:$REMOTE_DIR/runs/" cloud_artifacts/runs/
rsync -az "$REMOTE:$REMOTE_DIR/outputs/" cloud_artifacts/outputs/
rsync -az "$REMOTE:$REMOTE_DIR/data/synthetic/" cloud_artifacts/data_synthetic/
rsync -az "$REMOTE:$REMOTE_DIR/run_first5.log" cloud_artifacts/run_first5.log

echo "Saved remote artifacts in cloud_artifacts/"
