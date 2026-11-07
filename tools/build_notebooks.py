#!/usr/bin/env python
"""Compila los notebooks de ``notebooks/sources/*.py`` (formato percent) a ``.ipynb``.

Mantenemos la fuente de los notebooks como scripts Python con marcas
``# %%`` (el formato *percent* de jupytext): diffs limpios en git, syntax
highlighting en el editor y cero JSON a mano. Ejecutar::

    python tools/build_notebooks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "notebooks" / "sources"
OUTPUT = ROOT / "notebooks"


def parse_percent(source: str) -> list[tuple[str, str]]:
    """Separa un script en celdas ``("markdown" | "code", texto)``."""
    cells: list[tuple[str, str]] = []
    kind: str | None = None
    lines: list[str] = []

    for line in source.splitlines():
        stripped = line.strip()
        if stripped == "# %%":
            if kind:
                cells.append((kind, "\n".join(lines).strip("\n")))
            kind, lines = "code", []
        elif stripped.startswith("# %% [markdown]"):
            if kind:
                cells.append((kind, "\n".join(lines).strip("\n")))
            kind, lines = "markdown", []
        elif kind is not None:
            lines.append(line)
    if kind:
        cells.append((kind, "\n".join(lines).strip("\n")))
    return cells


def build_notebook(cells: list[tuple[str, str]]) -> dict:
    """Construye el dict nbformat 4.5 con las celdas dadas."""
    out_cells = []
    for kind, text in cells:
        source = text.splitlines(keepends=True)
        if kind == "markdown":
            out_cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
        else:
            out_cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": source,
                }
            )
    return {
        "cells": out_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    if not SOURCES.exists():
        print(f"No existe {SOURCES}")
        return 1
    OUTPUT.mkdir(exist_ok=True)
    built = 0
    for src in sorted(SOURCES.glob("*.py")):
        cells = parse_percent(src.read_text(encoding="utf-8"))
        if not cells:
            print(f"AVISO: {src.name} sin celdas; se ignora")
            continue
        nb = build_notebook(cells)
        dest = OUTPUT / f"{src.stem}.ipynb"
        dest.write_text(
            json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        print(f"OK  {dest.relative_to(ROOT)}  ({len(cells)} celdas)")
        built += 1
    print(f"{built} notebooks compilados")
    return 0


if __name__ == "__main__":
    sys.exit(main())
