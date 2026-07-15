# Verifier Bottleneck

Reproducible infrastructure for experiments studying how verifier quality limits
search and exploration. The repository is currently limited to initial project
infrastructure and CPU/GPU environment smoke tests.

## DataSphere smoke jobs

The job configurations use project `bt1k0f436ik4qgb96r7s` at submission time:

- `jobs/cpu-smoke.yaml`: CPU report on `c1.8`.
- `jobs/gpu-smoke.yaml`: CUDA-required report on `gt4.1`.

Install and authenticate the Yandex DataSphere CLI separately. Credentials and
tokens must stay outside this repository. The helper requires an explicit config
path and is a dry run unless `--execute` is also present. Review the command
without launching a job:

```bash
python scripts/submit_job.py jobs/cpu-smoke.yaml
```

After reviewing the config and obtaining budget approval, explicitly submit it:

```bash
python scripts/submit_job.py jobs/cpu-smoke.yaml --execute
```

The `--execute` form launches a paid cloud job. Repository checks and the helper's
default dry-run mode never submit jobs.

## GPU dependency timing smoke test

Choose the GPU explicitly by selecting a job config:

- `jobs/gpu-dependencies-t4-smoke.yaml`: T4 (`gt4.1`).
- `jobs/gpu-dependencies-v100-smoke.yaml`: V100 (`g1.1`).
- `jobs/gpu-dependencies-a100-smoke.yaml`: A100 (`g2.1`).
