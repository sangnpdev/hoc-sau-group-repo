"""Helpers for sharing code stored in notebooks with other notebooks."""

import json
from pathlib import Path
from typing import MutableMapping


def run_notebook_code(path: str | Path, namespace: MutableMapping[str, object]) -> None:
    """Execute code cells from a notebook in the caller's namespace.

    IPython's ``%run`` treats an .ipynb file as a Python script. Reading its
    notebook structure and compiling each code cell avoids that parse error.
    """
    notebook_path = Path(path)
    with notebook_path.open("r", encoding="utf-8") as file:
        notebook = json.load(file)

    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        raise ValueError(f"Not a valid nbformat 4 notebook: {notebook_path}")

    code_cells = []
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if source.strip():
            code_cells.append(f"# --- notebook cell {index} ---\n{source}")

    if code_cells:
        code = compile("\n\n".join(code_cells), str(notebook_path), "exec")
        exec(code, namespace)
