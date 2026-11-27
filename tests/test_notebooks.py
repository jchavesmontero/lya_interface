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


