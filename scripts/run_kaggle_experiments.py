#!/usr/bin/env python3
"""Submit the verifier-bottleneck experiments as a private Kaggle GPU kernel."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import textwrap
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KAGGLE_ROOT = Path("/Users/sergey/code/uzbeck_llm")
DEFAULT_REPO_URL = "https://github.com/nZiben/verifier-bottleneck-math.git"
DEFAULT_BRANCH = "experiment/game24-composition"
DEFAULT_SLUG = "verifier-bottleneck-experiments"


def kaggle_script(repo_url: str, branch: str, mode: str) -> str:
    return textwrap.dedent(
        f"""
        import datetime as dt
        import json
        import os
        import pathlib
        import shutil
        import subprocess


        REPO_URL = {repo_url!r}
        BRANCH = {branch!r}
        MODE = {mode!r}
        WORK = pathlib.Path("/kaggle/working")
        REPO_DIR = WORK / "verifier-bottleneck-math"
        ARTIFACTS = WORK / "artifacts"


        def run(cmd, cwd=None, env=None, display_cmd=None):
            print("$", display_cmd or cmd, flush=True)
            merged_env = os.environ.copy()
            if env:
                merged_env.update(env)
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=merged_env,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                print(line, end="", flush=True)
            code = proc.wait()
            if code:
                raise RuntimeError(f"command failed with exit code {{code}}: {{display_cmd or cmd}}")


        def optional_secret(name):
            try:
                from kaggle_secrets import UserSecretsClient
                return UserSecretsClient().get_secret(name)
            except Exception:
                return ""


        def clone_url():
            token = optional_secret("GITHUB_TOKEN")
            if token and REPO_URL.startswith("https://github.com/"):
                return REPO_URL.replace("https://", f"https://x-access-token:{{token}}@")
            return REPO_URL


        def copy_dir(src, dst):
            if src.exists():
                shutil.copytree(src, dst, dirs_exist_ok=True)


        hf_token = optional_secret("HF_TOKEN")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

        run("nvidia-smi || true")
        if REPO_DIR.exists():
            shutil.rmtree(REPO_DIR)
        run(
            f"git clone --branch {{BRANCH}} --single-branch {{clone_url()}} {{REPO_DIR}}",
            display_cmd=f"git clone --branch {{BRANCH}} --single-branch {{REPO_URL}} {{REPO_DIR}}",
        )
        run("git rev-parse HEAD", cwd=REPO_DIR)
        run("git status --short --branch", cwd=REPO_DIR)

        run("python -m pip install --upgrade pip")
        run("python -m pip uninstall -y torchao")
        run("python -m pip install --force-reinstall --no-cache-dir torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126")
        run("python -m pip install -e game24_composition -r game24_composition/requirements.txt", cwd=REPO_DIR)
        run("python -m pip install -r modp_verifier_sandbox/requirements.txt", cwd=REPO_DIR)

        game24_env = {{}}
        modp_env = {{}}
        if MODE == "smoke":
            game24_env.update({{
                "N_TRAIN_A": "80",
                "N_EVAL_A": "20",
                "N_TRAIN_B": "80",
                "N_EVAL_B": "20",
                "N_TEST_AB": "10",
                "EPOCHS": "1",
                "BATCH_SIZE": "1",
                "GRAD_ACCUM": "8",
            }})
            modp_env.update({{"EPOCHS": "5", "P": "17"}})
        elif MODE == "phase2_big":
            game24_env.update({{
                "MODEL_NAME": "Qwen/Qwen2.5-1.5B-Instruct",
                "DATA_DIR": "data/phase2_big",
                "OUTPUT_DIR": "outputs/phase2_big",
                "BASE_MODEL_DIR": ".cache/base_model_big",
                "B_RUN_DIR": "runs/phase2_big_b_only",
                "SEP_RUN_DIR": "runs/phase2_big_m_sep",
                "ST_RUN_DIR": "runs/phase2_big_self_train",
                "BATCH_SIZE": "1",
                "GRAD_ACCUM": "32",
            }})

        if MODE in {{"phase2", "phase2_big"}}:
            run("bash run_phase2.sh", cwd=REPO_DIR / "game24_composition", env=game24_env)
        else:
            run("bash run_first5.sh", cwd=REPO_DIR / "game24_composition", env=game24_env)
            run("bash run_modp_verifier.sh", cwd=REPO_DIR / "modp_verifier_sandbox", env=modp_env)

        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        copy_dir(REPO_DIR / "game24_composition" / "data", ARTIFACTS / "game24_composition" / "data")
        copy_dir(REPO_DIR / "game24_composition" / "runs", ARTIFACTS / "game24_composition" / "runs")
        copy_dir(REPO_DIR / "game24_composition" / "outputs", ARTIFACTS / "game24_composition" / "outputs")
        copy_dir(REPO_DIR / "modp_verifier_sandbox" / "checkpoints", ARTIFACTS / "modp_verifier_sandbox" / "checkpoints")
        copy_dir(REPO_DIR / "modp_verifier_sandbox" / "outputs", ARTIFACTS / "modp_verifier_sandbox" / "outputs")

        summary = [
            "# Kaggle Experiment Summary",
            "",
            f"- Finished UTC: {{dt.datetime.utcnow().isoformat(timespec='seconds')}}Z",
            f"- Mode: {{MODE}}",
            f"- Repo branch: {{BRANCH}}",
            f"- Game24 outputs: {{ARTIFACTS / 'game24_composition' / 'outputs'}}",
            f"- Game24 checkpoints: {{ARTIFACTS / 'game24_composition' / 'runs'}}",
        ]
        if MODE not in {{"phase2", "phase2_big"}}:
            summary.extend([
                f"- Mod-p outputs: {{ARTIFACTS / 'modp_verifier_sandbox' / 'outputs'}}",
                f"- Mod-p checkpoints: {{ARTIFACTS / 'modp_verifier_sandbox' / 'checkpoints'}}",
            ])
        summary.extend([
            "",
            "Phase 2 includes base-model eval, stronger B, composition re-check, perfect-checker self-training, and noisy-checker simulation." if MODE in {{"phase2", "phase2_big"}} else "No noisy Game24 checker, self-training, RL, GSM8K, or MATH run is included.",
        ])
        (ARTIFACTS / "kaggle_run_summary.md").write_text("\\n".join(summary) + "\\n", encoding="utf-8")

        manifest = sorted(str(path.relative_to(ARTIFACTS)) for path in ARTIFACTS.rglob("*") if path.is_file())
        (ARTIFACTS / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        shutil.make_archive(str(WORK / "verifier_bottleneck_artifacts"), "gztar", ARTIFACTS)
        print("Artifacts saved under", ARTIFACTS, flush=True)
        """
    ).strip() + "\n"


def configure_kaggle(kaggle_root: Path) -> tuple[Any, str]:
    config_dir = kaggle_root / ".kaggle"
    if not config_dir.exists():
        raise FileNotFoundError(f"Kaggle config directory not found: {config_dir}")

    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(config_dir))
    token_file = config_dir / "access_token"
    if token_file.exists() and not os.environ.get("KAGGLE_API_TOKEN"):
        os.environ["KAGGLE_API_TOKEN"] = str(token_file)
    if (config_dir / "credentials.json").exists() and not os.environ.get("KAGGLE_API_TOKEN"):
        os.environ.setdefault("HOME", str(kaggle_root))

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    username = api.get_config_value(api.CONFIG_NAME_USER)
    if not username:
        raise RuntimeError("Kaggle auth succeeded, but no username was returned.")
    return api, username


def stage(args: argparse.Namespace, username: str) -> Path:
    staging_dir = ROOT / ".kaggle_runs" / args.slug
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    code_file = "run_kaggle_experiments.py"
    (staging_dir / code_file).write_text(kaggle_script(args.repo_url, args.branch, args.mode), encoding="utf-8")
    metadata = {
        "id": f"{username}/{args.slug}",
        "title": args.title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (staging_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return staging_dir


def status_label(status: Any) -> str:
    if hasattr(status, "name"):
        return str(status.name).lower()
    if hasattr(status, "value"):
        return str(status.value).lower()
    return str(status).split(".")[-1].lower()


def wait_for_completion(api: Any, kernel_ref: str, poll_seconds: int) -> str:
    ok = {"complete", "kernelworkerstatus.complete"}
    bad = {"error", "failed", "cancel_acknowledged", "kernelworkerstatus.error", "kernelworkerstatus.cancel_acknowledged"}
    poll_errors = 0
    while True:
        try:
            response = api.kernels_status(kernel_ref)
            poll_errors = 0
        except Exception as exc:
            poll_errors += 1
            print(f"{kernel_ref}: status poll failed ({poll_errors}/10): {exc!r}", flush=True)
            if poll_errors >= 10:
                raise
            time.sleep(poll_seconds)
            continue
        label = status_label(response.status)
        print(f"{kernel_ref}: {label}", flush=True)
        if label in ok:
            return label
        if label in bad:
            message = getattr(response, "failure_message", "") or ""
            raise RuntimeError(f"{kernel_ref} ended with status {label}: {message}")
        time.sleep(poll_seconds)


def save_logs(api: Any, kernel_ref: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        log = api.kernels_logs(kernel_ref)
    except Exception as exc:
        log = f"Could not fetch Kaggle logs: {exc!r}\n"
    (output_dir / "kaggle-session.log").write_text(log or "", encoding="utf-8")


def download_outputs(api: Any, kernel_ref: str, output_dir: Path, file_pattern: str | None = None) -> int:
    import requests
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    compiled_pattern = re.compile(file_pattern) if file_pattern else None
    owner_slug, kernel_slug, _ = api.parse_kernel_string(kernel_ref)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    token = None
    while True:
        with api.build_kaggle_client() as kaggle:
            request = ApiListKernelSessionOutputRequest()
            request.user_name = owner_slug
            request.kernel_slug = kernel_slug
            request.page_size = 100
            if token:
                request.page_token = token
            response = kaggle.kernels.kernels_api_client.list_kernel_session_output(request)

        for item in response.files or []:
            if compiled_pattern and not compiled_pattern.search(item.file_name):
                continue

            outfile = output_dir / item.file_name
            for attempt in range(1, 4):
                try:
                    with requests.get(item.url, stream=True, timeout=(10, 60)) as result:
                        result.raise_for_status()
                        remote_size = int(result.headers.get("Content-Length", "0") or "0")
                        if outfile.exists() and remote_size and outfile.stat().st_size == remote_size:
                            print(f"{item.file_name}: already downloaded", flush=True)
                            break

                        outfile.parent.mkdir(parents=True, exist_ok=True)
                        part = outfile.with_suffix(outfile.suffix + ".part")
                        with part.open("wb") as handle:
                            for chunk in result.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    handle.write(chunk)
                        part.replace(outfile)
                        print(f"Output file downloaded to {outfile}", flush=True)
                        break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(5 * attempt)
            count += 1

        token = response.next_page_token
        if not token:
            break
    return count


def write_report(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = "\n\n" if path.exists() and path.stat().st_size else ""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(prefix + "\n".join(lines).rstrip() + "\n")


def append_downloaded_summary(lines: list[str], output_dir: Path, mode: str) -> None:
    if not mode.startswith("phase2"):
        return
    preferred_part = "phase2_big/phase2_summary.md" if mode == "phase2_big" else "phase2/phase2_summary.md"
    summaries = [path for path in sorted(output_dir.rglob("phase2_summary.md")) if preferred_part in str(path)]
    summaries = summaries or sorted(output_dir.rglob("phase2_summary.md"))
    if summaries:
        lines.extend(["", "### Downloaded Phase 2 Summary", "", summaries[0].read_text(encoding="utf-8").strip()])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["full", "smoke", "phase2", "phase2_big"], default="full")
    parser.add_argument("--kaggle-root", type=Path, default=DEFAULT_KAGGLE_ROOT)
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--title", default="Verifier Bottleneck Experiments")
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--timeout", type=int, default=43200)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--monitor-existing", action="store_true")
    parser.add_argument("--check", action="store_true", help="Check existing kernel status once and download outputs if complete.")
    parser.add_argument("--file-pattern", help="Regex for Kaggle output files to download.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "kaggle_results")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "kaggle_run_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    api, username = configure_kaggle(args.kaggle_root)
    kernel_ref = f"{username}/{args.slug}"
    output_dir = args.results_dir / args.slug

    lines = [
        "## Второй этап экспериментов" if args.mode.startswith("phase2") else "## Kaggle Run",
        "",
        f"- Started UTC: {started}",
        f"- Kernel: `{kernel_ref}`",
        f"- Branch: `{args.branch}`",
        f"- Mode: `{args.mode}`",
        f"- Local output directory: `{output_dir}`",
    ]

    try:
        if args.check:
            response = api.kernels_status(kernel_ref)
            status = status_label(response.status)
            lines.extend(["- Action: checked existing kernel", f"- Status: `{status}`"])
            if status in {"complete", "kernelworkerstatus.complete"}:
                save_logs(api, kernel_ref, output_dir)
                count = download_outputs(api, kernel_ref, output_dir, args.file_pattern)
                lines.append(f"- Downloaded files: `{count}`")
                if args.file_pattern:
                    lines.append(f"- File pattern: `{args.file_pattern}`")
                append_downloaded_summary(lines, output_dir, args.mode)
            write_report(args.report, lines)
            print(f"{kernel_ref}: {status}", flush=True)
            return

        if not args.monitor_existing:
            staging_dir = stage(args, username)
            print(f"Staged Kaggle kernel at {staging_dir}", flush=True)
            response = api.kernels_push(str(staging_dir), timeout=str(args.timeout), acc=args.accelerator)
            if response is None or getattr(response, "error", ""):
                raise RuntimeError(f"Kaggle push failed: {getattr(response, 'error', response)}")
            lines.append(f"- Pushed version: `{getattr(response, 'versionNumber', 'unknown')}`")
            if getattr(response, "url", ""):
                lines.append(f"- Kaggle URL: {response.url}")
        else:
            lines.append("- Action: monitored existing kernel")

        if args.no_wait:
            lines.append("- Status: submitted, not waited")
            write_report(args.report, lines)
            return

        status = wait_for_completion(api, kernel_ref, args.poll_seconds)
        save_logs(api, kernel_ref, output_dir)
        downloaded_count = download_outputs(api, kernel_ref, output_dir, args.file_pattern)
        lines.extend([
            f"- Status: `{status}`",
            f"- Downloaded files: `{downloaded_count}`",
            "",
            "Downloaded outputs are under `kaggle_results/`; the remote run also creates `verifier_bottleneck_artifacts.tar.gz`.",
        ])
        append_downloaded_summary(lines, output_dir, args.mode)
        write_report(args.report, lines)
    except Exception as exc:
        save_logs(api, kernel_ref, output_dir)
        lines.extend(["", f"- Status: failed or blocked: `{exc!r}`"])
        write_report(args.report, lines)
        raise


if __name__ == "__main__":
    main()
