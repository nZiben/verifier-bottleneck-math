#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
SEED="${SEED:-42}"

DATA_DIR="${DATA_DIR:-data/synthetic}"
RUN_DIR="${RUN_DIR:-runs/m_sep}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

N_TRAIN_A="${N_TRAIN_A:-3000}"
N_EVAL_A="${N_EVAL_A:-300}"
N_TRAIN_B="${N_TRAIN_B:-3000}"
N_EVAL_B="${N_EVAL_B:-300}"
N_TEST_AB="${N_TEST_AB:-300}"

EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-768}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"

mkdir -p "$DATA_DIR" "$RUN_DIR" "$OUTPUT_DIR"

echo "== Self check =="
"$PYTHON_BIN" -m game24_composition.self_check

echo "== Generate data =="
"$PYTHON_BIN" -m game24_composition.generate_data \
  --out_dir "$DATA_DIR" \
  --seed "$SEED" \
  --n_train_A "$N_TRAIN_A" \
  --n_eval_A "$N_EVAL_A" \
  --n_train_B "$N_TRAIN_B" \
  --n_eval_B "$N_EVAL_B" \
  --n_test_AB "$N_TEST_AB"

echo "== Train M_sep, keeping all epoch checkpoints =="
"$PYTHON_BIN" -m game24_composition.train_sft \
  --model_name "$MODEL_NAME" \
  --train_file "$DATA_DIR/train_sep.jsonl" \
  --output_dir "$RUN_DIR" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum "$GRAD_ACCUM" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --seed "$SEED"

echo "== Composition gap eval =="
"$PYTHON_BIN" -m game24_composition.evaluate \
  --model_path "$RUN_DIR" \
  --splits "$DATA_DIR/eval_A.jsonl" "$DATA_DIR/eval_B.jsonl" "$DATA_DIR/test_AB.jsonl" \
  --out "$OUTPUT_DIR/composition_gap_results.json" \
  --temperature 0.0 \
  --max_new_tokens 256 \
  --seed "$SEED"

echo "== Baseline pass@k =="
"$PYTHON_BIN" -m game24_composition.evaluate \
  --model_path "$RUN_DIR" \
  --splits "$DATA_DIR/test_AB.jsonl" \
  --out "$OUTPUT_DIR/baseline_passk.json" \
  --pass_k 1,4,16,64,256 \
  --num_samples 256 \
  --temperature 0.7 \
  --top_p 0.95 \
  --max_new_tokens 256 \
  --seed "$SEED"

echo "== Exploration sweep =="
"$PYTHON_BIN" -m game24_composition.sweep_exploration \
  --model_path "$RUN_DIR" \
  --test_file "$DATA_DIR/test_AB.jsonl" \
  --out_dir "$OUTPUT_DIR/exploration_sweep" \
  --seed "$SEED"

echo "== Done =="
echo "Model and checkpoints: $RUN_DIR"
find "$RUN_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort
echo "Outputs: $OUTPUT_DIR"
