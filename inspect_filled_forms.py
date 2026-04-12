"""
inspect_filled_forms.py — After generating a forms_*.zip, run this to see
exactly which fields got overlaid on which pages.

Usage:
    python3 inspect_filled_forms.py /path/to/forms_NYC-XXXXX/
"""
from __future__ import annotations

import sys
from pathlib import Path
import pdfplumber


def _extract_words_set(page):
    return {(round(w["x0"]), round(w["top"]), w["text"])
            for w in page.extract_words()}


def inspect(filled_dir: Path):
    orig_map = {
        "snap": Path("samples/forms/ldss_4826_snap.pdf"),
        "medicaid": Path("samples/forms/doh_4220_medicaid.pdf"),
    }

    for kind, orig in orig_map.items():
        matches = list(filled_dir.glob(f"{kind}_*.pdf"))
        if not matches:
            continue
        filled = matches[0]
        print(f"\n{'=' * 60}")
        print(f"  {kind.upper()}  →  {filled.name}")
        print(f"{'=' * 60}")

        if not orig.exists():
            print(f"  (original {orig} missing — skip diff)")
            continue

        total = 0
        with pdfplumber.open(orig) as o, pdfplumber.open(filled) as f:
            for page_idx in range(min(len(o.pages), len(f.pages))):
                o_set = _extract_words_set(o.pages[page_idx])
                f_words = f.pages[page_idx].extract_words()
                added = [w for w in f_words
                         if (round(w["x0"]), round(w["top"]), w["text"]) not in o_set]
                if not added:
                    continue
                total += len(added)
                print(f"\n  Page {page_idx + 1}:  {len(added)} overlay words")
                for w in added:
                    print(f"    at ({w['x0']:6.1f}, {w['top']:6.1f})  →  '{w['text']}'")

        print(f"\n  Total overlay words across all pages: {total}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 inspect_filled_forms.py /path/to/forms_NYC-XXXX/")
        sys.exit(1)
    d = Path(sys.argv[1])
    if not d.is_dir():
        print(f"Not a directory: {d}")
        sys.exit(1)
    inspect(d)
