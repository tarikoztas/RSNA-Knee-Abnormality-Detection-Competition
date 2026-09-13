"""`# %%` hucre isaretli bir .py dosyasini .ipynb'ye cevirir.

Kullanim:
    python tools/py2ipynb.py notebooks/00_phase0_dicom_survey.py

Jupytext'e bagimli olmamak icin yazildi — Kaggle'a import edilebilir minimal
nbformat v4 JSON uretir.

Ek direktif — bir kaynak dosyayi notebook hucresine gomer:

    #!writefile src/data/dicom_io.py /kaggle/working/dicom_io.py

Bu satiri iceren hucre, ilk satiri `%%writefile <hedef>` olan ve govdesi kaynak
dosyanin tam icerigi olan bir hucreye donusur. Boylece modulun tek bir dogru
surumu olur (repo'daki dosya), notebook onu Kaggle'da diske yazar.
"""
import json
import sys
from pathlib import Path


def split_cells(text):
    """Kaynagi (kind, satirlar) hucrelerine ayir."""
    cells, kind, buf = [], "code", []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# %%"):
            if buf:
                cells.append((kind, buf))
            kind = "markdown" if "[markdown]" in stripped else "code"
            buf = []
        else:
            buf.append(line)
    if buf:
        cells.append((kind, buf))
    return cells


def to_source(kind, lines):
    """Markdown hucrelerinde bas yorum isaretlerini soy, bos kenarlari kirp."""
    if kind == "markdown":
        out = []
        for ln in lines:
            if ln.startswith("# "):
                out.append(ln[2:])
            elif ln.strip() == "#":
                out.append("")
            else:
                out.append(ln)
        lines = out
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return [ln + "\n" for ln in lines[:-1]] + lines[-1:] if lines else []


def expand_writefile(lines, base_dir):
    """`#!writefile <kaynak> <hedef>` satirini kaynak dosyanin icerigiyle degistir.

    Yollar once repo kokune (base_dir'in ustu), sonra base_dir'e gore cozulur —
    boylece direktif `notebooks/x.py` icinden `src/...` diye yazilabilir.
    """
    out = []
    for line in lines:
        s = line.strip()
        if not s.startswith("#!writefile "):
            out.append(line)
            continue
        parts = s.split()
        if len(parts) != 3:
            raise ValueError(f"gecersiz direktif: {s}")
        _, rel_src, dest = parts
        for root in (base_dir.parent, base_dir):
            cand = root / rel_src
            if cand.exists():
                break
        else:
            raise FileNotFoundError(f"{rel_src} bulunamadi ({base_dir} icinden)")
        out.append(f"%%writefile {dest}")
        out.extend(cand.read_text(encoding="utf-8").splitlines())
    return out


def main(src_path):
    src = Path(src_path)
    text = src.read_text(encoding="utf-8")

    cells = []
    for kind, lines in split_cells(text):
        lines = expand_writefile(lines, src.parent)
        source = to_source(kind, lines)
        if not source:
            continue
        cell = {"cell_type": kind, "metadata": {}, "source": source}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    dst = src.with_suffix(".ipynb")
    dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    n_md = sum(c["cell_type"] == "markdown" for c in cells)
    print(f"{dst}  ->  {len(cells)} hucre ({n_md} markdown, {len(cells)-n_md} kod)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("kullanim: python tools/py2ipynb.py <dosya.py>")
    main(sys.argv[1])
