"""Convert `# %%` percent-format scripts into executed .ipynb notebooks.

Deliberately dependency-free: builds the notebook JSON directly and executes
cells in-process, so it needs neither nbformat nor nbconvert/jupyter.

    python experiments/_py2nb.py                 # convert + execute all
    python experiments/_py2nb.py 01              # just the 01_* source
    python experiments/_py2nb.py 01 --no-exec    # convert only
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELL_RE = re.compile(r"^# %%(?P<md>\s*\[markdown\])?\s*$")


def split_cells(source: str) -> list[tuple[str, str]]:
    cells, kind, buf = [], "code", []
    for line in source.splitlines():
        m = CELL_RE.match(line)
        if m:
            if buf:
                cells.append((kind, "\n".join(buf).strip("\n")))
            kind = "markdown" if m.group("md") else "code"
            buf = []
        else:
            buf.append(line)
    if buf:
        cells.append((kind, "\n".join(buf).strip("\n")))
    return [(k, t) for k, t in cells if t.strip()]


def to_source_list(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        pass
    return lines


def build_notebook(cells: list[tuple[str, str]]) -> dict:
    nb_cells = []
    for kind, text in cells:
        if kind == "markdown":
            text = "\n".join(re.sub(r"^# ?", "", ln) for ln in text.splitlines()).strip()
            nb_cells.append({
                "cell_type": "markdown", "metadata": {},
                "source": to_source_list(text),
            })
        else:
            nb_cells.append({
                "cell_type": "code", "execution_count": None, "metadata": {},
                "outputs": [], "source": to_source_list(text),
            })
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def execute(nb: dict, workdir: Path, out_path: Path | None = None) -> bool:
    """Run code cells in one shared namespace, capturing stdout/stderr.

    The notebook is flushed to disk after every cell, so a long run that is
    interrupted still leaves the completed cells and their outputs on disk.
    """
    ns: dict = {"__name__": "__main__"}
    count = 0
    ok = True
    prev = sys.path[:]
    sys.path.insert(0, str(workdir))

    def flush():
        if out_path is not None:
            out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        count += 1
        cell["execution_count"] = count
        src = "".join(cell["source"])
        buf = io.StringIO()
        outputs = []
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                exec(compile(src, f"<cell {count}>", "exec"), ns)
        except Exception:
            text = buf.getvalue()
            if text:
                outputs.append({"output_type": "stream", "name": "stdout",
                                "text": to_source_list(text)})
            tb = traceback.format_exc()
            outputs.append({
                "output_type": "error", "ename": "Error", "evalue": "see traceback",
                "traceback": tb.splitlines(),
            })
            cell["outputs"] = outputs
            flush()
            print(f"    cell {count} FAILED:\n{tb[-2500:]}")
            ok = False
            break
        text = buf.getvalue()
        if text:
            outputs.append({"output_type": "stream", "name": "stdout",
                            "text": to_source_list(text)})
        cell["outputs"] = outputs
        flush()
        print(f"    cell {count} ok")

    sys.path[:] = prev
    return ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("which", nargs="?", default=None)
    p.add_argument("--no-exec", action="store_true")
    args = p.parse_args()

    sources = sorted((HERE / "src").glob("*.py"))
    if args.which:
        sources = [s for s in sources if s.stem.startswith(args.which)]
    if not sources:
        print("no matching sources")
        return

    project_root = HERE.parent
    for src in sources:
        cells = split_cells(src.read_text(encoding="utf-8"))
        nb = build_notebook(cells)
        out = HERE / f"{src.stem}.ipynb"
        print(f"{src.name}: {len(cells)} cells", flush=True)
        out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        if not args.no_exec:
            print("  executing ...", flush=True)
            ok = execute(nb, project_root, out)
            print("  ok" if ok else "  finished with errors")
        print(f"  wrote {out.name}")


if __name__ == "__main__":
    main()
