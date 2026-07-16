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
The setup script installs into the current Jupyter kernel, does not launch a Job,
and does not download a model. Restart the kernel after installation.

Use `notebooks/01_experiment_template.ipynb` when starting an experiment. It
initializes deterministic seeds, records the Git commit and hardware, remains
disabled by default, and writes results under the ignored `outputs/` directory.

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

## GPU dependency timing smoke test

Choose the GPU explicitly by selecting a job config:

- `jobs/gpu-dependencies-t4-smoke.yaml`: T4 (`gt4.1`).
- `jobs/gpu-dependencies-v100-smoke.yaml`: V100 (`g1.1`).
- `jobs/gpu-dependencies-a100-smoke.yaml`: A100 (`g2.1`).

These Jobs rebuild or deploy their Python environment for each standalone run.
They are retained for batch compatibility, not as a dependency cache.
