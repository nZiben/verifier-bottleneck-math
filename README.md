# Verifier Bottleneck

Reproducible infrastructure for experiments studying how verifier quality limits
search and exploration. The repository is currently limited to initial project
infrastructure and CPU/GPU environment smoke tests.

## Local setup

Python 3.10 is the supported runtime. From the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell, create and activate the environment with:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

PyTorch is deliberately not a project dependency. If it is already installed,
the report includes its version and CUDA details. Otherwise, the CPU report still
runs successfully.

## System report CLI

Write a report to standard output:

```bash
verifier-bottleneck-report
```

Write a report to a JSON file:

```bash
verifier-bottleneck-report --output reports/system.json
```

Require a usable CUDA device (returns a non-zero status when CUDA is unavailable):

```bash
verifier-bottleneck-report --require-cuda
```

The module form works without installing the console script when `src` is on
`PYTHONPATH`:

```bash
python -m verifier_bottleneck.cli --help
```

## Local checks

The test suite requires no GPU and no network access:

```bash
pytest -q
ruff check .
mypy src
```

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

The dependency benchmark adapts `PRIME-RL/RL-Compositionality` to DataSphere's
managed Python environment. It does not copy the upstream Dockerfile. Each job's
manual `env.python` section selects Python 3.10 and the common
`requirements-rl-compositionality-gpu.txt` environment. It uploads
`src/verifier_bottleneck` as a non-empty local module because the current
DataSphere SDK cannot validate an empty manual `local-paths` list.

Choose the GPU explicitly by selecting a job config:

- `jobs/gpu-dependencies-t4-smoke.yaml`: T4 (`gt4.1`).
- `jobs/gpu-dependencies-v100-smoke.yaml`: V100 (`g1.1`).
- `jobs/gpu-dependencies-a100-smoke.yaml`: A100 (`g2.1`).

The job installs and imports the upstream dependency surface without downloading
a model or logging into Weights & Biases. `flash-attn` is excluded because the
standard package is architecture-specific and upstream installs it separately
with `--no-build-isolation`. Keeping it out makes the dependency comparison
consistent across T4, V100, and A100. The original CPU and GPU system smoke jobs
remain unchanged.

The dependency requirements file intentionally contains only package specifiers.
The DataSphere CLI parser does not accept the comment lines that ordinary `pip`
requirements files allow. Dependency provenance and rationale are documented
here instead of inside that file.

The adaptation references these upstream `main` blobs:

- `requirements.txt`: `fe6536f7e40a8c549033e98597bb34395daee364`
- `pyproject.toml`: `c015cc011312b28abc51ec5d2b751e840f361812`
- `Dockerfile`: `18c0727a0d40c71db45ad5b3a9acd4ad2090e2ca`

Review the submission safely:

```powershell
$config = "jobs/gpu-dependencies-t4-smoke.yaml"
python scripts/submit_job.py $config
```

After explicit budget approval, measure the end-to-end cold run while preserving
DataSphere's live logs:

```powershell
$timer = [System.Diagnostics.Stopwatch]::StartNew()
python scripts/submit_job.py $config --execute
$jobExitCode = $LASTEXITCODE
$timer.Stop()
$timer.Elapsed
"Job exit code: $jobExitCode"
```

The elapsed stopwatch includes client upload, environment preparation, and job
execution. The selected GPU's dependency report records individual import
durations, versions, failures, and CUDA/GPU details. DataSphere may reuse a cached
environment on later runs, so distinguish the first cold run from subsequent
warm runs.
