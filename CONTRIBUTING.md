# Contributing

This repository separates research protocols from model implementations. New
work should extend the owning layer instead of copying an existing end-to-end
experiment.

## Repository map

| Change | Location |
| --- | --- |
| Deterministic dataset construction | `src/verifier_bottleneck/data/` |
| Model-independent decoding and sampling | `src/verifier_bottleneck/evaluation/` |
| Typed config and experiment orchestration | `src/verifier_bottleneck/experiments/` |
| Models and backend-specific optimization | `src/verifier_bottleneck/training/` |
| Recorded result payloads, reports, and plots | `src/verifier_bottleneck/analysis/` |
| Thin DataSphere process entry points | `src/verifier_bottleneck/jobs/` |
| Versioned experiment parameters | `configs/` |

For Countdown A/B/A+B, the reusable protocol is in
`experiments/countdown_protocol.py`, grammar-constrained temperature evaluation
is in `evaluation/countdown.py`, and the shared run lifecycle is in
`experiments/countdown_runtime.py`. Scratch and Qwen trainers contain only
backend-specific encoding, models, and optimization. Qwen prompt encoding and
decoding have separate modules so either can change independently.

## Adding a dataset

1. Implement a deterministic split builder under `data/`.
2. Add typed fields and strict YAML parsing under `experiments/`.
3. Return explicit train, validation, and held-out test collections.
4. Test seed reproducibility, split disjointness, and verifier correctness.

## Adding a model backend

1. Add a backend config under `experiments/` and its implementation under
   `training/`.
2. Reuse the protocol's dataset builder, evaluation sweep, result payload, and
   experiment lifecycle.
3. Keep the job entry point thin; it should only execute and export the run.
4. Do not introduce a second result schema or temperature-sweep implementation.

Create a new protocol rather than adding ambiguous mode flags when the skills,
curriculum, or evaluation question changes.

## Quality checks

Use Python 3.10 and deterministic seeds. Before merging, run:

```powershell
python -m pytest -q
python -m ruff check src tests scripts
python -m mypy src
```

Never commit datasets, checkpoints, credentials, model caches, or generated run
archives. Cloud jobs and large model downloads require explicit approval.
