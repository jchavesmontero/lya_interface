"""Validate tracked text notebooks without relying on ignored local .ipynb files."""
import ast
from pathlib import Path

import jupytext
import nbformat
import pytest


NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


@pytest.mark.parametrize("path", sorted(NOTEBOOKS.glob("*.py")), ids=lambda p: p.stem)
def test_notebook_structure_and_code(path):
    notebook = jupytext.read(path)
    nbformat.validate(notebook)
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["language"] == "python"
    assert notebook["metadata"]["jupytext"]["formats"] == "ipynb,py:percent"
    assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for cell in code_cells:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        # IPython line/cell magics (%autoreload, %%time) are not Python syntax.
        source = "".join(cell["source"])
        source = "\n".join("" if line.lstrip().startswith("%") else line for line in source.splitlines())
        ast.parse(source, filename=str(path))
    code = "\n".join("".join(c["source"]) for c in code_cells)
    assert "model.close()" in code or "fit.close()" in code
    restored = jupytext.reads(jupytext.writes(notebook, fmt="py:percent"), fmt="py:percent")
    assert [(c.cell_type, c.source) for c in restored.cells] == [
        (c.cell_type, c.source) for c in notebook.cells
    ]


def test_joint_notebook_uses_dr2_vega_configuration_and_fit_workflow():
    notebook = jupytext.read(NOTEBOOKS / "joint_DR1P1D_DR2BAO.py")
    source = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    tree = ast.parse("\n".join(line for line in source.splitlines()
                               if not line.lstrip().startswith("%")))
    strings = {node.value for node in ast.walk(tree)
               if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    configs = {s for s in strings if s.startswith("joint_dr1p1d_forestmpggp_dr2bao_lya")}
    assert {"joint_dr1p1d_forestmpggp_dr2bao_lya_kaiser_fixed_beta",
            "joint_dr1p1d_forestmpggp_dr2bao_lya_kaiser_free_beta"} <= configs
    for name in configs:
        assert (NOTEBOOKS.parent / "examples" / f"{name}.yaml").is_file()
        if "_kaiser_" in name:
            assert (NOTEBOOKS.parent / "examples" / f"{name}_rmin10.yaml").is_file()
    for call in ("Fit(CONFIG, metal_alpha_reference=START_FROM)", "fit.minimize(", "fit.save(",
                 "Fit.from_results(", "fit.plot_triangle(", "global_ic_point(",
                 "default_ic_path(info, IC_VARIATION)", "vega.metal_alpha_at("):
        assert call in source


def test_cup1d_vega_mock_notebook_has_compact_joint_workflow():
    notebook = jupytext.read(NOTEBOOKS / "04_joint_cup1d_vega_mock.py")
    source = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert "examples/joint_desi_dr1_vega_mock.yaml" in source
    assert "igm_tau_eff_3" in source
    assert "parameters_for_evaluation" in source
    assert "plot_vega_point" in source
    assert "minimize_point" in source
