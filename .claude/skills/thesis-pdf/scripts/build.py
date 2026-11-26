"""Compile the thesis with Typst and check the PDF against the measurable format rules.

    uv run --no-project --with typst --with pymupdf python .claude/skills/thesis-pdf/scripts/build.py

Defaults: thesis/src/main.typ -> thesis/build/thesis.pdf, with thesis/ as the Typst root.
Exit code 0 when it compiled and no hard rule failed (warnings are printed), 1 otherwise.
"""

import argparse
from collections import Counter
from pathlib import Path
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

CM = 72 / 2.54
A4 = (595.28, 841.89)
MARGIN = dict(left=3 * CM, right=2 * CM, top=2.5 * CM, bottom=3 * CM)
MARKERS = re.compile(r"\[(?:TO FILL|TO CONFIRM|TO ADD|PENDING)[^\]]*\]?")
# File paths must not appear in the PDF (author rule, 2026-11-24). URLs are allowed.
PATHS = re.compile(r"(?<![\w/:.])(?:[\w.-]+/)+[\w.-]*|\b[\w-]+\.(?:py|json|jsonl|toml|md|typ|safetensors|png|svg)\b")


def text_lines(page):
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"]).strip()
            if text:
                yield line["bbox"], text, line["spans"]


def check(pdf_path):
    import pymupdf

    document = pymupdf.open(pdf_path)
    failures, warnings = [], []
    fonts, gaps = Counter(), Counter()
    labels = [page.get_label() or str(index + 1) for index, page in enumerate(document)]
    # Body pages are the Arabic-numbered pages after the last Roman-numbered front-matter page.
    roman = [index for index, label in enumerate(labels) if re.fullmatch(r"[ivxlc]+", label)]
    body_start = roman[-1] + 1 if roman else 0
    markers, margin_pages, paths = [], [], []
    references_label = abstract_words = None
    for index, page in enumerate(document):
        label = labels[index]
        width, height = page.rect.width, page.rect.height
        if abs(width - A4[0]) > 1 or abs(height - A4[1]) > 1:
            failures.append(f"page {index + 1} is {width:.0f} x {height:.0f} pt, not A4")
        lines = list(text_lines(page))
        text = "\n".join(entry[1] for entry in lines)
        markers.extend((label, found) for found in MARKERS.findall(text))
        if index >= 2:  # skip cover and title page
            paths.extend((label, found) for found in PATHS.findall(re.sub(r"https?://\S+", "", text)))
        if references_label is None and any(entry[1] == "REFERENCES" for entry in lines):
            references_label = label
        if any(entry[1] == "ABSTRACT" for entry in lines):
            body = text.split("ABSTRACT", 1)[1].split("Keywords", 1)[0]
            abstract_words = len(body.split())
        if index < body_start:
            continue  # front matter: cover frames and centred pages are laid out by the template
        footer_top = height - MARGIN["bottom"]
        body_lines = [entry for entry in lines if entry[0][1] < footer_top - 1]
        previous = None
        for (x0, y0, x1, y1), _, spans in body_lines:
            for span in spans:
                fonts[(span["font"], round(span["size"], 1))] += len(span["text"])
            baseline = spans[0]["origin"][1]
            if previous is not None and 10 < baseline - previous < 30:
                gaps[round(baseline - previous, 1)] += 1
            previous = baseline
        if body_lines:
            left = min(entry[0][0] for entry in body_lines)
            right = max(entry[0][2] for entry in body_lines)
            top = min(entry[0][1] for entry in body_lines)
            bottom = max(entry[0][3] for entry in body_lines)
            # Justified lines overhang the right margin by a few points of trailing space.
            if (left < MARGIN["left"] - 3 or right > width - MARGIN["right"] + 6
                    or top < MARGIN["top"] - 6 or bottom > footer_top + 6):
                margin_pages.append(label)
    print(f"Pages: {len(labels)} total, {body_start} front matter, {len(labels) - body_start} numbered")
    if references_label and references_label.isdigit():
        body_pages = int(references_label) - 1
        print(f"Body pages before the references: {body_pages} (rule: at least about 50 to 75, "
              "not counting figures, tables, references and appendices)")
        if body_pages < 50:
            warnings.append(f"body is {body_pages} pages; the rule asks for about 50 to 75")
    else:
        warnings.append("no REFERENCES heading found; add the bibliography")
    if fonts:
        (font, size), _ = fonts.most_common(1)[0]
        print(f"Body font: {font} {size} pt (rule: 13 pt)")
        if abs(size - 13) > 0.2:
            failures.append(f"dominant body size is {size} pt, not 13 pt")
    if gaps:
        gap = gaps.most_common(1)[0][0]
        print(f"Line pitch: {gap} pt (rule: 1.3 lines, about 19.4 pt)")
        if not 18.4 <= gap <= 20.4:
            warnings.append(f"line pitch {gap} pt is outside 18.4 to 20.4 pt")
    if margin_pages:
        warnings.append("text outside the margins on page(s) " + ", ".join(margin_pages[:20])
                        + " (wide table, figure or unbreakable line)")
    if abstract_words is not None:
        print(f"Abstract: {abstract_words} words (rule: about 360)")
        if abstract_words > 420:
            warnings.append(f"abstract has {abstract_words} words; the rule is about 360")
    if markers:
        print(f"Unfilled markers: {len(markers)}")
        for label, found in markers[:60]:
            print(f"  p.{label}: {found[:100]}")
    pending = [entry for entry in markers if "PENDING" in entry[1]]
    if pending:
        failures.append(f"{len(pending)} [PENDING] marker(s) in the PDF; move every gap to thesis/pending.md")
    if paths:
        print(f"File-path-like strings: {len(paths)}")
        for label, found in paths[:40]:
            print(f"  p.{label}: {found}")
        warnings.append("file paths must not appear in the PDF; check the strings above (model and dataset IDs such as org/name are allowed)")
    for message in warnings:
        print(f"WARNING: {message}")
    for message in failures:
        print(f"FAIL: {message}")
    return not failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", default="thesis/src/main.typ")
    parser.add_argument("output", nargs="?", default="thesis/build/thesis.pdf")
    parser.add_argument("--root", default=None, help="Typst root; default is the folder above the source folder")
    args = parser.parse_args(argv)
    import typst

    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    root = Path(args.root).resolve() if args.root else source.parent.parent
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        typst.compile(str(source), output=str(output), root=str(root))
    except Exception as error:
        print(f"FAIL: Typst did not compile:\n{error}", file=sys.stderr)
        return 1
    print(f"Compiled: {output}")
    return 0 if check(output) else 1


if __name__ == "__main__":
    raise SystemExit(main())
