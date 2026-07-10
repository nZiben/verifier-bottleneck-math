#!/usr/bin/env bash
# Colab install script for verifier-bottleneck-math.
#
# Install ORDER matters: torch -> vllm==0.8.3 -> flash-attn -> verl -> ray.
# vLLM (and, less predictably, verl's own transitive deps) can silently
# replace the installed torch/vllm build if installed out of order or without
# --no-deps guards, so every step re-checks torch's version afterward.
#
# Idempotent: safe to re-run on the same runtime — each step skips reinstalling
# a package that's already at the pinned version.
set -euo pipefail

TORCH_PIN="2.6.0"
VLLM_PIN="0.8.3"
VERL_PIN="0.8.0"

log() { echo "== $* =="; }

pkg_version() {
  # Prints installed version of $1, or empty string if not installed.
  python -c "import importlib.metadata as m; print(m.version('$1'))" 2>/dev/null || true
}

torch_version() { pkg_version torch; }
vllm_version() { pkg_version vllm; }

assert_unchanged() {
  # assert_unchanged <label> <before> <after>
  local label="$1" before="$2" after="$3"
  if [[ "$before" != "$after" ]]; then
    echo "ERROR: $label changed from $before to $after during this step." >&2
    echo "This usually means a package silently pulled in an incompatible torch/vllm build." >&2
    exit 1
  fi
}

log "Checking Python version"
PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
echo "python: $PYTHON_VERSION"
if [[ "$PYTHON_VERSION" != "3.10" ]]; then
  echo "ERROR: expected Python 3.10, found $PYTHON_VERSION" >&2
  exit 1
fi

log "Step 1/7: checking CUDA via nvidia-smi"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found — this runtime has no GPU attached." >&2
  echo "In Colab: Runtime -> Change runtime type -> Hardware accelerator -> GPU." >&2
  exit 1
fi
nvidia-smi
CUDA_VERSION_DETECTED="$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -n1)"
if [[ -z "$CUDA_VERSION_DETECTED" ]]; then
  echo "ERROR: could not parse a CUDA version out of nvidia-smi output." >&2
  exit 1
fi
echo "Detected driver CUDA version: $CUDA_VERSION_DETECTED"
CUDA_OK="$(python -c "print(1 if tuple(map(int, '$CUDA_VERSION_DETECTED'.split('.'))) >= (12, 1) else 0)")"
if [[ "$CUDA_OK" != "1" ]]; then
  echo "ERROR: CUDA $CUDA_VERSION_DETECTED < 12.1, which verl/vLLM require." >&2
  echo "In Colab: request an A100/L4 runtime, or Runtime -> Factory reset runtime and retry." >&2
  exit 1
fi

log "Step 2/7: installing torch==$TORCH_PIN matching detected CUDA"
# torch 2.6.0 ships prebuilt wheels for cu124 and cu126 (no cu121 wheel exists).
# Forward-compat: a driver reporting CUDA >= the wheel's toolkit version can run it.
CUDA_MAJOR_MINOR_OK_126="$(python -c "print(1 if tuple(map(int, '$CUDA_VERSION_DETECTED'.split('.'))) >= (12, 6) else 0)")"
if [[ "$CUDA_MAJOR_MINOR_OK_126" == "1" ]]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cu126"
else
  TORCH_INDEX="https://download.pytorch.org/whl/cu124"
  echo "NOTE: driver CUDA $CUDA_VERSION_DETECTED < 12.6 — using cu124 torch wheels." \
       "This needs driver >= 12.4 to run; verify if the import check at the end fails." >&2
fi
CURRENT_TORCH="$(torch_version)"
if [[ "$CURRENT_TORCH" == "$TORCH_PIN"* ]]; then
  echo "torch==$CURRENT_TORCH already installed, skipping."
else
  pip install "torch==$TORCH_PIN" --index-url "$TORCH_INDEX"
fi
TORCH_VERSION_AFTER_TORCH="$(torch_version)"
echo "torch version: $TORCH_VERSION_AFTER_TORCH"

log "Step 3/7: installing vllm==$VLLM_PIN"
CURRENT_VLLM="$(vllm_version)"
if [[ "$CURRENT_VLLM" == "$VLLM_PIN" ]]; then
  echo "vllm==$VLLM_PIN already installed, skipping."
else
  pip install "vllm==$VLLM_PIN"
fi
assert_unchanged "torch version" "$TORCH_VERSION_AFTER_TORCH" "$(torch_version)"
echo "vllm version: $(vllm_version)"

log "Step 4/7: installing flash-attn --no-build-isolation"
if python -c "import flash_attn" >/dev/null 2>&1; then
  echo "flash-attn already installed ($(pkg_version flash-attn)), skipping."
else
  pip install flash-attn --no-build-isolation
fi
assert_unchanged "torch version" "$TORCH_VERSION_AFTER_TORCH" "$(torch_version)"
assert_unchanged "vllm version" "$VLLM_PIN" "$(vllm_version)"

log "Step 5/7: installing verl==$VERL_PIN"
CURRENT_VERL="$(pkg_version verl)"
if [[ "$CURRENT_VERL" == "$VERL_PIN" ]]; then
  echo "verl==$VERL_PIN already installed, skipping."
else
  pip install "verl==$VERL_PIN"
  if [[ "$(torch_version)" != "$TORCH_VERSION_AFTER_TORCH" || "$(vllm_version)" != "$VLLM_PIN" ]]; then
    echo "WARNING: installing verl moved torch/vllm (torch: $TORCH_VERSION_AFTER_TORCH -> $(torch_version)," \
         "vllm: $VLLM_PIN -> $(vllm_version)). Reinstalling verl with --no-deps and restoring pins." >&2
    pip install "verl==$VERL_PIN" --no-deps --force-reinstall
    pip install "torch==$TORCH_PIN" --index-url "$TORCH_INDEX" --force-reinstall --no-deps
    pip install "vllm==$VLLM_PIN" --force-reinstall --no-deps
    # verl's own non-torch/vllm deps (from its requirements.txt) that --no-deps skipped.
    pip install "numpy>=2.0.0" "pyarrow>=19.0.0" "tensordict>=0.8.0,<=0.10.0,!=0.9.0" "transformers!=5.6.0" "packaging>=20.0"
  fi
fi
assert_unchanged "torch version" "$TORCH_VERSION_AFTER_TORCH" "$(torch_version)"
assert_unchanged "vllm version" "$VLLM_PIN" "$(vllm_version)"

log "Step 6/7: installing ray[default]"
if python -c "import ray" >/dev/null 2>&1; then
  echo "ray already installed ($(pkg_version ray)), skipping."
else
  pip install "ray[default]"
fi
assert_unchanged "torch version" "$TORCH_VERSION_AFTER_TORCH" "$(torch_version)"
assert_unchanged "vllm version" "$VLLM_PIN" "$(vllm_version)"

log "Step 7/7: final sanity check"
python -c "
import torch, vllm, verl
print('torch:', torch.__version__, '| cuda available:', torch.cuda.is_available())
print('vllm :', vllm.__version__)
print('verl :', verl.__version__)
"

log "Done — all pins verified, nothing was silently downgraded"
