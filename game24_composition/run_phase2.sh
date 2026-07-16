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

DATA_DIR="${DATA_DIR:-data/phase2}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/phase2}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-.cache/base_model}"
B_RUN_DIR="${B_RUN_DIR:-runs/phase2_b_only}"
SEP_RUN_DIR="${SEP_RUN_DIR:-runs/phase2_m_sep}"
ST_RUN_DIR="${ST_RUN_DIR:-runs/phase2_self_train}"

N_TRAIN_A="${N_TRAIN_A:-3000}"
N_EVAL_A="${N_EVAL_A:-100}"
N_TRAIN_B="${N_TRAIN_B:-8000}"
N_EVAL_B="${N_EVAL_B:-100}"
N_TEST_AB="${N_TEST_AB:-100}"

B_EPOCHS="${B_EPOCHS:-4}"
SEP_EPOCHS="${SEP_EPOCHS:-4}"
ST_EPOCHS="${ST_EPOCHS:-2}"
LR="${LR:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-768}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
MAX_ACCEPTED_PER_TASK="${MAX_ACCEPTED_PER_TASK:-4}"

mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$B_RUN_DIR" "$SEP_RUN_DIR" "$ST_RUN_DIR"

echo "== Self check =="
"$PYTHON_BIN" -m game24_composition.self_check

echo "== Cache base model once =="
"$PYTHON_BIN" -m game24_composition.cache_model \
  --model_name "$MODEL_NAME" \
  --out_dir "$BASE_MODEL_DIR"

echo "== Generate phase 2 data =="
"$PYTHON_BIN" -m game24_composition.generate_data \
  --out_dir "$DATA_DIR" \
  --seed "$SEED" \
  --n_train_A "$N_TRAIN_A" \
  --n_eval_A "$N_EVAL_A" \
  --n_train_B "$N_TRAIN_B" \
  --n_eval_B "$N_EVAL_B" \
  --n_test_AB "$N_TEST_AB"

echo "== Base model before SFT =="
"$PYTHON_BIN" -m game24_composition.evaluate \
  --model_path "$BASE_MODEL_DIR" \
  --splits "$DATA_DIR/eval_A.jsonl" "$DATA_DIR/eval_B.jsonl" "$DATA_DIR/test_AB.jsonl" \
  --out "$OUTPUT_DIR/base_model_results.json" \
  --temperature 0.0 \
  --max_new_tokens 256 \
  --seed "$SEED"
"$PYTHON_BIN" -m game24_composition.evaluate \
  --model_path "$BASE_MODEL_DIR" \
  --splits "$DATA_DIR/test_AB.jsonl" \
  --out "$OUTPUT_DIR/base_model_passk.json" \
  --pass_k 1,4,16 \
  --num_samples 16 \
  --temperature 0.7 \
  --top_p 0.95 \
  --max_new_tokens 256 \
  --seed "$SEED"

echo "== Train B-only =="
"$PYTHON_BIN" -m game24_composition.train_sft \
  --model_name "$BASE_MODEL_DIR" \
  --train_file "$DATA_DIR/train_B.jsonl" \
  --output_dir "$B_RUN_DIR" \
  --epochs "$B_EPOCHS" \
  --lr "$LR" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum "$GRAD_ACCUM" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --seed "$SEED"
"$PYTHON_BIN" -m game24_composition.evaluate \
  --model_path "$B_RUN_DIR" \
  --splits "$DATA_DIR/eval_B.jsonl" \
  --out "$OUTPUT_DIR/b_only_results.json" \
  --temperature 0.0 \
  --max_new_tokens 256 \
  --seed "$SEED"

echo "== Train stronger A+B =="
"$PYTHON_BIN" -m game24_composition.train_sft \
  --model_name "$BASE_MODEL_DIR" \
  --train_file "$DATA_DIR/train_sep.jsonl" \
  --output_dir "$SEP_RUN_DIR" \
  --epochs "$SEP_EPOCHS" \
  --lr "$LR" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum "$GRAD_ACCUM" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --seed "$SEED"
"$PYTHON_BIN" -m game24_composition.evaluate \
  --model_path "$SEP_RUN_DIR" \
  --splits "$DATA_DIR/eval_A.jsonl" "$DATA_DIR/eval_B.jsonl" "$DATA_DIR/test_AB.jsonl" \
  --out "$OUTPUT_DIR/strong_sep_results.json" \
  --temperature 0.0 \
  --max_new_tokens 256 \
  --seed "$SEED"
"$PYTHON_BIN" -m game24_composition.evaluate \
  --model_path "$SEP_RUN_DIR" \
  --splits "$DATA_DIR/test_AB.jsonl" \
  --out "$OUTPUT_DIR/strong_sep_passk.json" \
  --pass_k 1,4,16,32 \
  --num_samples 32 \
  --temperature 0.7 \
  --top_p 0.95 \
  --max_new_tokens 256 \
  --seed "$SEED"
echo "== Perfect-checker self-training =="
"$PYTHON_BIN" -m game24_composition.collect_accepted_ab \
  --examples "$DATA_DIR/test_AB.jsonl" \
  --generations "$OUTPUT_DIR/strong_sep_passk_generations.jsonl" \
  --out "$OUTPUT_DIR/accepted_ab_train.jsonl" \
  --stats_out "$OUTPUT_DIR/accepted_ab_train.stats.json" \
  --max_per_task "$MAX_ACCEPTED_PER_TASK"
"$PYTHON_BIN" -m game24_composition.train_sft \
  --model_name "$SEP_RUN_DIR" \
  --train_file "$OUTPUT_DIR/accepted_ab_train.jsonl" \
  --output_dir "$ST_RUN_DIR" \
  --epochs "$ST_EPOCHS" \
  --lr "$LR" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum "$GRAD_ACCUM" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --seed "$SEED" \
  --allow_ab
"$PYTHON_BIN" -m game24_composition.evaluate \
  --model_path "$ST_RUN_DIR" \
  --splits "$DATA_DIR/eval_A.jsonl" "$DATA_DIR/eval_B.jsonl" "$DATA_DIR/test_AB.jsonl" \
  --out "$OUTPUT_DIR/self_train_results.json" \
  --temperature 0.0 \
  --max_new_tokens 256 \
  --seed "$SEED"
"$PYTHON_BIN" -m game24_composition.evaluate \
  --model_path "$ST_RUN_DIR" \
  --splits "$DATA_DIR/test_AB.jsonl" \
  --out "$OUTPUT_DIR/self_train_passk.json" \
  --pass_k 1,4,16 \
  --num_samples 16 \
  --temperature 0.7 \
  --top_p 0.95 \
  --max_new_tokens 256 \
  --seed "$SEED"

echo "== Noisy checker simulation =="
"$PYTHON_BIN" -m game24_composition.noisy_checker_eval \
  --generations "$OUTPUT_DIR/strong_sep_passk_generations.jsonl" \
  --out_json "$OUTPUT_DIR/noisy_checker_eval.json" \
  --out_csv "$OUTPUT_DIR/noisy_checker_eval.csv" \
  --seed "$SEED"

echo "== Phase 2 summary =="
"$PYTHON_BIN" -m game24_composition.summarize_phase2 \
  --data_dir "$DATA_DIR" \
  --outputs_dir "$OUTPUT_DIR" \
  --out "$OUTPUT_DIR/phase2_summary.md"

echo "== Done =="
echo "Outputs: $OUTPUT_DIR"
