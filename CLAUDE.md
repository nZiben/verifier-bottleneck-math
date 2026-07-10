# CLAUDE.md

## Project

**The Verifier Bottleneck — Can an AI Teach Itself Mathematics?**

A research project studying self-improvement loops (generate → verify → fine-tune → repeat)
in math LLMs. The central framing is **two dials**:

1. **Checker reliability (α, β)** — α = how often the checker accepts a correct answer,
   β = how often it accepts a wrong one. The loop only improves when α > β
   (i.e. the checker carries real information about correctness, I(c;V) > 0).
2. **Exploration budget** — sampling diversity / search / KL control. Decides whether the
   loop merely *sharpens* existing ability (Yue regime) or *expands* into genuinely new
   ability by recombining latent skills (ProRL regime).

Goal: prove the two-dial law, map the {collapse, sharpen, expand} phase diagram, and catch
skill creation on real math (composition sandbox + GSM8K/MATH).

## Tech stack

- **verl** (Volcano Engine RL for LLMs) integrated as a **pip dependency**.
- Experiments run on **Google Colab** — likely Colab Pro (A100 / L4).
- The smoke test (`notebooks/00_setup_smoke_test.ipynb`) targets **Colab Pro (L4/A100)
  first**. Free T4 is a later size-down (drop to a 0.5B model) if Pro isn't available,
  not the default target — a 1.5B model plus vLLM's KV cache is tight on a T4's 16 GB.

## Hard constraints (verl install)

- **Python 3.10, CUDA ≥ 12.1.**
- **Install ORDER matters:** torch → `vllm==0.8.3` → flash-attn → verl.
  vLLM will silently override the installed torch if the order is wrong — always assert
  torch was not downgraded after installing vllm.
- Use `VLLM_USE_V1=1` at runtime.
- On single-GPU Colab, vLLM (KV cache) and FSDP (training weights) share one GPU — keep
  `gpu_memory_utilization` low (~0.4) and set `enforce_eager=False`.

## Repo layout

```
requirements.txt
notebooks/00_setup_smoke_test.ipynb   # Colab entry point
configs/                              # verl YAML / hydra overrides
scripts/colab_install.sh
src/vbm/
  rewards/    # α,β-noisy checker + I(c;V) logging  (Dial 1)
  sandbox/    # synthetic composition task
  metrics/    # pass@1, pass@k, held-out composition eval
  loops/      # thin wrappers over verl trainers
experiments/
```

## Working agreement

- **Work step by step.** After each step, stop and let me review before continuing.
- **This repo runs on Colab, not my laptop.** Do NOT install heavy deps (torch, vllm,
  flash-attn, verl) locally in this VS Code environment.
- **Do not invent verl's API.** verl changes fast and may be newer than your training data.
  When unsure of a config key or function signature, inspect the installed package or
  verl's docs and cite the actual file path — tell me what to verify rather than guessing.
- **Commit after every green step** before moving on, so a broken later step never costs a
  working earlier one.
