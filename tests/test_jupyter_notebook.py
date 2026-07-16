import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def test_setup_notebook_is_clean_and_safe() -> None:
    notebook_path = PROJECT_ROOT / "notebooks" / "00_datasphere_setup.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert notebook["nbformat"] == 4
    assert all(cell.get("execution_count") is None for cell in notebook["cells"])
    assert all(not cell.get("outputs") for cell in notebook["cells"])
    assert "scripts/setup_jupyter.py" in code
    assert "RUN_INSTALLATION = False" in code
    assert "INSTALL_GPU_DEPENDENCIES = False" in code
    assert '["git", "pull", "--ff-only"]' in code
    assert "datasphere project job" not in code
    assert "from_pretrained" not in code
    assert "D:\\\\" not in code


def test_experiment_template_is_clean_and_disabled() -> None:
    notebook_path = PROJECT_ROOT / "notebooks" / "01_experiment_template.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert all(cell.get("execution_count") is None for cell in notebook["cells"])
    assert all(not cell.get("outputs") for cell in notebook["cells"])
    assert "SEED = 42" in code
    assert "RUN_EXPERIMENT = False" in code
    assert 'Path("outputs")' in code
    assert '["git", "rev-parse", "HEAD"]' in code
    assert "from_pretrained" not in code
    assert "datasphere project job" not in code
    assert "D:\\\\" not in code
