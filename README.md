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
