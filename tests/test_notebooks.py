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
        ast.parse("".join(cell["source"]), filename=str(path))
    assert "model.close()" in "\n".join("".join(c["source"]) for c in code_cells)
    restored = jupytext.reads(jupytext.writes(notebook, fmt="py:percent"), fmt="py:percent")
    assert [(c.cell_type, c.source) for c in restored.cells] == [
        (c.cell_type, c.source) for c in notebook.cells
    ]


def test_joint_notebook_uses_combined_configuration_and_native_diagnostics():
    notebook = jupytext.read(NOTEBOOKS / "03_joint_p1d_desidr2_lya_bao.py")
    source = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    tree = ast.parse(source)
    strings = {node.value for node in ast.walk(tree)
               if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert "examples/cup1d_desidr2_lya_bao.yaml" in strings
    assert "bao.desi_dr2.desi_bao_lya" in strings
    calls = [node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert calls.count("get_model") == 1
    assert calls.count("plot_point") == 2
    assert calls.count("minimize_point") == 1
    assert "RUN_MINIMIZATION = False" in source


def test_cup1d_vega_mock_notebook_has_compact_joint_workflow():
    notebook = jupytext.read(NOTEBOOKS / "04_joint_cup1d_vega_mock.py")
    source = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert "examples/joint_desi_dr1_vega_mock.yaml" in source
    assert "igm_tau_eff_3" in source
    assert "parameters_for_evaluation" in source
    assert "plot_vega_point" in source
    assert "minimize_point" in source
