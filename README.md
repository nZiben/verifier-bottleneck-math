# Verifier Bottleneck

Reproducible infrastructure for experiments studying how verifier quality limits
search and exploration. DataSphere JupyterLab is the primary interactive
development environment; standalone Jobs remain available for approved batch
runs.

## DataSphere JupyterLab quick start

Open DataSphere project `bt1k0f436ik4qgb96r7s` in JupyterLab. Clone this
repository once using **Git → Clone a Repository**:

```text
https://github.com/nZiben/verifier-bottleneck-math.git
```

Open `notebooks/00_datasphere_setup.ipynb`. Its installation cell starts in
dry-run mode. Set:

```python
RUN_INSTALLATION = True
INSTALL_DEV_TOOLS = True
INSTALL_GPU_DEPENDENCIES = True
```

Use the GPU option only when the full RL/vLLM dependency environment is needed.
The first notebook cell immediately adds the checkout's `src/` directory to the
current kernel, so `import verifier_bottleneck` works without installation or a
restart. The setup script is only needed for missing third-party dependencies
or persistent CLI entry points. It does not launch a Job or download a model.
Restart only when pip replaces a package already imported by the kernel.

Use `notebooks/01_experiment_template.ipynb` when starting an experiment. It
initializes deterministic seeds, records the Git commit and hardware, remains
disabled by default, and writes results under the ignored `outputs/` directory.

## First sandbox experiment

`notebooks/02_modular_addition_sandbox.ipynb` is the first implemented research
experiment. It:

- creates the complete addition table modulo a prime without downloading data;
- trains a 1- or 2-layer causal transformer from scratch on a deterministic
  portion of that table;
- treats residues as abstract tokens in equations of the form `[a, +, b, =]`;
- applies configurable verifier operating points
  `alpha = P(V=1 | c=1)` and `beta = P(V=1 | c=0)`;
- compares empirical accepted accuracy with
  `a' = alpha*a / (alpha*a + beta*(1-a))`;
- writes JSON and a notebook-generated plot under the ignored `outputs/`
  directory.

The formula defines accuracy conditional on the verifier accepting a proposal:
`a' = P(c=1 | V=1)`. It is undefined when the verifier accepts nothing.

Start with the roughly 14,000-parameter smoke configuration:

```python
from pathlib import Path

from verifier_bottleneck.experiments.modular_addition import (
    describe_experiment,
    load_experiment_config,
    run_from_path,
)

config_path = Path("configs/sandbox/modular_addition_smoke.yaml")
config = load_experiment_config(config_path)
print(describe_experiment(config))
result = run_from_path(config_path)
print(result["run_id"])
print(result["record_path"])
print(result["summary_path"])
```

This call runs in the current Python process: it does not use a subprocess,
launch a DataSphere Job, download a model, or create a cloud resource.
`configs/sandbox/modular_addition_paper.yaml` is a separate 100,000-step,
2-layer, width-128 configuration inspired by the original grokking setup. Do
not use it as the first smoke test.

If the repository is not installed in the current notebook kernel, install only
the lightweight project/config layer:

```python
%pip install -e ".[sandbox]"
```

The sandbox extra intentionally does not install PyTorch because its wheel is
large. Use the PyTorch already supplied by Colab or the DataSphere notebook
image. Validate the config without importing PyTorch:

```bash
python -m verifier_bottleneck.experiments.modular_addition \
  --config configs/sandbox/modular_addition_smoke.yaml \
  --dry-run
```

Run from a terminal only after the dry run is reviewed:

```bash
python -m verifier_bottleneck.experiments.modular_addition \
  --config configs/sandbox/modular_addition_smoke.yaml
```

Reinstalling the editable project after pulling the updated `pyproject.toml`
also exposes the shorter `verifier-bottleneck-mod-add` command.

### Saved run format

Every execution creates a unique directory below the configured output root:

```text
outputs/<experiment-name>/<timestamp>_<config-hash>/
    record.json
    summary.json
    metrics.jsonl
    config.yaml
    source-manifest.json
    accuracy-and-verifier.png
```

`record.json` is the canonical paper record. It includes:

- run ID, status, UTC timestamps, duration, description, and tags;
- exact parsed configuration and a configuration hash;
- Git commit and hashes for source, configs, tests, notebooks, and requirements;
- Python, OS, installed package versions, PyTorch, CUDA, GPU model and memory;
- all exposed random seeds and deterministic PyTorch settings;
- dataset construction, split sizes, and train/test dataset fingerprints;
- architecture, parameter counts, dtype, optimizer and hyperparameters;
- every evaluation checkpoint and verifier operating point;
- final and best metrics, runtime, peak GPU memory, and artifact hashes;
- exception information when a run fails.

Only a small allowlist of runtime context is saved. Credentials, tokens, and the
complete process environment are never written.

Build comparison-ready tables after one or more runs:

```bash
python -m verifier_bottleneck.experiment_comparison \
  --root outputs \
  --output-directory outputs/comparison
```

This writes `runs.csv`, `verifier-measurements.csv`, and `comparison.json`.
They can be loaded directly by pandas for paper tables and figures.

The equivalent terminal command inside DataSphere is:

```bash
python scripts/setup_jupyter.py --dev --gpu
```

For a lightweight CPU/kernel setup:

```bash
python scripts/setup_jupyter.py --dev
```

To update an existing checkout:

```bash
git status --short
git pull --ff-only
```

Clone only once. Reinstall only when `pyproject.toml` or a requirements file
changes. See `docs/jupyterlab.md` for the full workflow, cost controls, and
troubleshooting.

## Local Windows setup

From a fresh PowerShell:

```powershell
Set-Location "<path-to-verifier-bottleneck-math>"
& ".\.venv\Scripts\Activate.ps1"
python -m pip install -e ".[dev]"
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

## Modular-addition sandbox Job

`jobs/modular-addition-sandbox-smoke.yaml` runs the 14,000-parameter sandbox on
CPU `c1.4`. It uses a separate minimal environment containing only pinned
PyYAML and CPU-only PyTorch. It does not install the full RL/vLLM dependency
surface and does not require a GPU.

The Job returns three files:

- `modular-addition-sandbox-smoke-record.json`: canonical paper record;
- `modular-addition-sandbox-smoke-summary.json`: comparison-ready summary;
- `modular-addition-sandbox-smoke.zip`: the complete run directory, including
  checkpoint metrics, exact config and source manifest.

Preview the submission command without launching anything:

```powershell
python scripts/submit_job.py jobs/modular-addition-sandbox-smoke.yaml
```

After reviewing the config and receiving budget approval, launch it explicitly:

```powershell
python scripts/submit_job.py jobs/modular-addition-sandbox-smoke.yaml --execute
```

The second command creates a paid DataSphere Job. Dependency deployment is
standalone Job overhead even though the actual experiment is very small.

## Paper-scale modular-addition Job

`jobs/modular-addition-paper.yaml` runs the separate paper-scale configuration
on one T4 (`gt4.1`) with CUDA. It trains the prime-97, two-layer, width-128
model for 100,000 steps and exports the same complete experiment record as the
smoke Job. This is a real experiment run, not a smoke test, and its compute
cost should be reviewed before submission.

Preview the exact DataSphere command without launching the Job:

```powershell
python scripts/submit_job.py jobs/modular-addition-paper.yaml
```

Only after manual budget approval, launch it explicitly:

```powershell
python scripts/submit_job.py jobs/modular-addition-paper.yaml --execute
```

The Job returns `modular-addition-paper-record.json`,
`modular-addition-paper-summary.json`, and `modular-addition-paper.zip`.
The ZIP contains the checkpoint metrics, exact configuration, source manifest,
and all other registered run artifacts. A single run is not sufficient for a
paper claim; use additional reviewed configurations and seeds for replication.

## GPU dependency timing smoke test

Choose the GPU explicitly by selecting a job config:

- `jobs/gpu-dependencies-t4-smoke.yaml`: T4 (`gt4.1`).
- `jobs/gpu-dependencies-v100-smoke.yaml`: V100 (`g1.1`).
- `jobs/gpu-dependencies-a100-smoke.yaml`: A100 (`g2.1`).

These Jobs rebuild or deploy their Python environment for each standalone run.
They are retained for batch compatibility, not as a dependency cache.
