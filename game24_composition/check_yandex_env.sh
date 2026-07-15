#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

echo "Yandex URL: ${YANDEX_CLOUD_URL:-not set}"

if [ -n "${YANDEX_LOGIN:-}" ]; then
  echo "Yandex login: set"
else
  echo "Yandex login: not set"
fi

if [ -n "${YANDEX_PASSWORD:-}" ]; then
  echo "Yandex password: set"
else
  echo "Yandex password: not set"
fi

if command -v yc >/dev/null 2>&1; then
  echo "yc CLI: $(command -v yc)"
  yc config list
else
  echo "yc CLI: not installed"
fi

if [ -n "${GPU_HOST:-}" ]; then
  echo "GPU host: set"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "${GPU_SSH_USER:-ubuntu}@${GPU_HOST}" 'hostname && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
else
  echo "GPU host: not set"
fi
