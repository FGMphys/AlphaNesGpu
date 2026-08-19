#!/usr/bin/env python3
"""Sprint 2: rewrite STAF-CG CUDA/C++ sources to use staf_real / STAF_TF_DTYPE."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path("/home/francegm/AlphaNesGpu/STAF-CG/src")
EXTS = {".cc", ".cu", ".h", ".cuh"}


def is_source(p: Path) -> bool:
    if p.suffix in EXTS:
        return True
    # reforce.cu.cc
    return p.name.endswith(".cu.cc")


def convert_text(text: str, path: Path) -> str:
    if "staf_real.h" not in text:
        if path.suffix == ".h" or path.suffix == ".cuh":
            m = re.search(r"(#define \w+\s*\n)", text)
            if m:
                text = text[: m.end()] + '#include "staf_real.h"\n' + text[m.end() :]
            else:
                text = '#include "staf_real.h"\n' + text
        else:
            # After last tensorflow include if present, else after the first include block.
            lines = text.splitlines(keepends=True)
            last_tf = -1
            last_inc = -1
            for i, line in enumerate(lines):
                if line.startswith("#include"):
                    last_inc = i
                    if "tensorflow" in line:
                        last_tf = i
            insert_at = (last_tf if last_tf >= 0 else last_inc) + 1
            if insert_at <= 0:
                text = '#include "staf_real.h"\n' + text
            else:
                lines.insert(insert_at, '#include "staf_real.h"\n')
                text = "".join(lines)

    text = text.replace(': double"', ': " STAF_TF_DTYPE')
    text = re.sub(r"\bdouble\b", "real", text)

    def math_sub(fn: str, staf: str, s: str) -> str:
        return re.sub(rf"(?<!staf_)(?<!\w){fn}\(", f"{staf}(", s)

    text = math_sub("exp", "staf_exp", text)
    text = math_sub("cos", "staf_cos", text)
    text = math_sub("sin", "staf_sin", text)
    text = math_sub("sqrt", "staf_sqrt", text)
    text = math_sub("pow", "staf_pow", text)

    # float suffixes used as kernel literals (A1/A2)
    text = re.sub(r"\b([0-9]+\.[0-9]*)f\b", r"real(\1)", text)
    text = re.sub(r"\b([0-9]+)f\b", r"real(\1)", text)
    return text


def main() -> None:
    n = 0
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or not is_source(p):
            continue
        if "mixture" in p.parts:
            continue
        orig = p.read_text()
        new = convert_text(orig, p)
        if new != orig:
            p.write_text(new)
            n += 1
            print("converted", p.relative_to(ROOT))
    print(f"updated {n} files")


if __name__ == "__main__":
    main()
