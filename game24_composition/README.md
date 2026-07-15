# Game24 Composition Sandbox

First five experimental steps for testing whether separate symbolic decoding and numeric Game24 skills compose.

This package implements only:

1. A controllable symbolic Game24 sandbox.
2. SFT on separate A-only decoding and B-only numeric Game24 data.
3. A/B/AB evaluation to measure the composition gap.
4. Baseline empirical pass@k on AB tasks.
5. An exploration sweep with a perfect checker.

It does not implement noisy checkers, self-training rounds, RL, GSM8K, MATH, tree search, majority vote, or step-by-step verifiers.

## Setup

Run commands from this directory:

```bash
cd game24_composition
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The default small model is `Qwen/Qwen2.5-0.5B-Instruct`.

## Run Everything

```bash
bash run_first5.sh
```

The script runs data generation, SFT, greedy A/B/AB evaluation, AB pass@k, and the exploration sweep. Training saves epoch checkpoints under `runs/m_sep/checkpoint-*`; checkpoints are not pruned unless you pass `--save_total_limit` directly to `train_sft.py`.

## Cloud Env

```bash
cp .env.example .env
bash check_yandex_env.sh
```

`.env` is ignored by git. Do not commit cloud credentials.

If Yandex gives you a GPU VM hostname, set `GPU_HOST` and run:

```bash
bash run_remote_gpu.sh
```

It copies this package to the VM, runs `run_first5.sh`, and pulls back `runs/`, `outputs/`, `data/synthetic/`, and `run_first5.log` into `cloud_artifacts/`.

## 1. Generate Data

```bash
python -m game24_composition.generate_data \
  --out_dir data/synthetic \
  --seed 42 \
  --n_train_A 3000 \
  --n_eval_A 300 \
  --n_train_B 3000 \
  --n_eval_B 300 \
  --n_test_AB 300
```

This writes:

- `data/synthetic/train_A.jsonl`
- `data/synthetic/eval_A.jsonl`
- `data/synthetic/train_B.jsonl`
- `data/synthetic/eval_B.jsonl`
- `data/synthetic/test_AB.jsonl`
- `data/synthetic/train_sep.jsonl`
- `data/synthetic/manifest.json`

`train_sep.jsonl` is exactly A-only plus B-only data. It contains no AB composition examples. Numeric Game24 tuple pools are split by canonical sorted tuple, and `train_B` is disjoint from `test_AB`.

## 2. Train M_sep

```bash
python -m game24_composition.train_sft \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --train_file data/synthetic/train_sep.jsonl \
  --output_dir runs/m_sep \
  --epochs 3 \
  --lr 2e-4 \
  --batch_size 4 \
  --grad_accum 8 \
  --max_seq_len 768 \
  --lora_r 16 \
  --lora_alpha 32 \
  --seed 42
```

Training uses chat-style formatting:

```text
User:
{question}

Assistant:
{answer}
```

## 3. Confirm Composition Gap

```bash
python -m game24_composition.evaluate \
  --model_path runs/m_sep \
  --splits data/synthetic/eval_A.jsonl data/synthetic/eval_B.jsonl data/synthetic/test_AB.jsonl \
  --out outputs/composition_gap_results.json \
  --temperature 0.0 \
  --max_new_tokens 256 \
  --seed 42
```

This writes:

- `outputs/composition_gap_results.json`
- `outputs/composition_gap_generations.jsonl`

Metrics are reported separately as `decode_accuracy`, `numeric_game24_pass@1`, and `symbolic_game24_pass@1`.

## 4. Baseline pass@k on AB

```bash
python -m game24_composition.evaluate \
  --model_path runs/m_sep \
  --splits data/synthetic/test_AB.jsonl \
  --out outputs/baseline_passk.json \
  --pass_k 1,4,16,64,256 \
  --num_samples 256 \
  --temperature 0.7 \
  --top_p 0.95 \
  --max_new_tokens 256 \
  --seed 42
```

This writes:

- `outputs/baseline_passk.json`
- `outputs/baseline_passk_generations.jsonl`

pass@k is empirical: each task is solved at k if any of the first k generated samples passes the perfect checker.

## 5. Exploration Sweep

```bash
python -m game24_composition.sweep_exploration \
  --model_path runs/m_sep \
  --test_file data/synthetic/test_AB.jsonl \
  --out_dir outputs/exploration_sweep \
  --seed 42
```

This writes:

- `outputs/exploration_sweep/results.csv`
- `outputs/exploration_sweep/results.json`
- `outputs/exploration_sweep/generations.jsonl`
- `outputs/exploration_sweep/passk_curve.png`
- `outputs/first5_summary.md`

## Quick Core Check

This does not load a model:

```bash
python -m game24_composition.self_check
```
