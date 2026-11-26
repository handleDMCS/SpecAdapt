---
name: thesis-pdf
description: Write and build the SpecAdapt graduation thesis PDF (English, UET/VNU Hanoi format) from thesis/skeleton.md. Use when asked to draft, extend, format, rebuild or check the thesis PDF, a chapter of it, its figures or its references.
---

# Thesis PDF

Produces `thesis/build/thesis.pdf` from Typst sources in `thesis/src/`. Content comes from `thesis/skeleton.md`; layout comes from this skill.

## Read first

1. `thesis/skeleton.md`: the content tree, the status of every section and the evidence file behind every number.
2. [format-rules.md](format-rules.md): the format rules, their sources and the open points.

## Grounding rules

These outrank everything else in this skill.

- Write a fact only if `thesis/skeleton.md` lists it with an evidence file. Open that file and re-read the number before typing it. If the file and the skeleton disagree, stop and report; do not pick one.
- A section marked `BLANK` is left out of the PDF and listed in `thesis/pending.md`. No estimates, no numbers carried over from another pair or setup, no "expected" results.
- A section marked `PARTIAL` gets the supported part only; the missing part goes to `thesis/pending.md`.
- Never put `#pending[...]`, `[PENDING]` or any other gap marker in the PDF. Every gap (missing section, missing citation, missing evidence file, open check) is an entry in `thesis/pending.md`, with the skeleton section it belongs to and the evidence it needs.
- Never fill a front-matter field (title, author, supervisor, major, degree, year, acknowledgement) that the skeleton marks `[TO FILL]` or `[TO CONFIRM]`. Leave the template placeholder.
- Cite only works in the skeleton's Literature table that have a complete, checked BibTeX entry in `thesis/src/refs.bib`. A sentence that needs a citation you do not have is left out of the PDF and listed in `thesis/pending.md`. Never write a reference from memory. The project's GitHub repository (https://github.com/handleDMCS/SpecAdapt) is always in the references and is cited where the thesis says the code is available.
- Keep validation results labelled as validation, replay results as replay, and the October 4-bit demo apart from the November runs, exactly as the skeleton separates them.
- State the limits the skeleton lists (its "Limits to state", "Caveat" and "Do not claim" lines) next to the result they qualify.
- The thesis is in English. Neutral technical tone; no "I", "we" or "you" in the main text.

## Writing style

Set by the author on 2026-11-24. These outrank the template and the skeleton's evidence pointers.

- **No file paths anywhere in the PDF:** not in the body, captions, tables, footnotes or appendices. No repository paths, script names, result-file names or job files. Cite the GitHub repository once instead (see the grounding rules). Code identifiers (class or function names) are also kept out of the prose; describe what the component does.
- **Prose first, numbers in structured blocks.** The text explains the idea and the pattern intuitively, in natural language, and points to a table, chart or figure for the values. Do not string numbers through a paragraph. A sentence may carry one key number when it is the point of the sentence (the abstract may give the main results); everything else goes in a table or figure.
- Explain mechanisms before details: what problem a component solves, how it works in plain words, then its exact settings in a table.
- Captions describe the content and the setting (pair, prompts, decoding); they never name the file a number came from.
- Run IDs, Git commits and pinned revisions belong only in the reproducibility appendix tables.

## Chapter structure

The thesis uses the standard English thesis structure, chosen by the author. Do not rename, merge, split or reorder chapters; `thesis/skeleton.md` lists the sections inside each one.

| Part | What goes in it |
| --- | --- |
| Front matter | cover, title page, authorship, supervisor's approval, acknowledgement, abstract, table of contents, lists of figures, tables and algorithms, abbreviations |
| 1. Introduction | motivation, problem and research question, contributions, thesis overview |
| 2. Literature review | a full chapter, written from sources only |
| 3. Methodology | a full chapter: the method, then the experimental design |
| 4 and 5. Results | what was measured and what the tables show |
| 6. Discussion | what the results mean, relation to prior work, threats to validity |
| 7. Conclusion and future work | main results with an assessment, then what remains |
| References | IEEE style |
| Appendices | reproducibility, full tables, environment, prompts and splits |

Results and Discussion are separate on purpose. A results chapter reports numbers and reads the tables; it does not explain them. The discussion interprets, and every sentence in it points back to a table or figure in Chapters 4 and 5. A cause that no experiment tested is left out or labelled as a hypothesis.

The authorship and approval pages are not in the generic structure; the university template requires them, so they stay.

## Files

```text
thesis/
├── skeleton.md          content and evidence (read-only for this skill unless asked to update it)
├── pending.md           every gap left out of the PDF: section, what is missing, evidence needed
├── sample/              the three format references
├── src/
│   ├── main.typ         front-matter fields, chapter includes, bibliography, appendix
│   ├── uet-thesis.typ   layout, copied from this skill's template/
│   ├── chapters/        one .typ file per chapter: 1-introduction.typ ... 7-conclusion.typ
│   └── refs.bib         checked BibTeX entries only
├── src/diagrams.typ     CeTZ diagrams (flowcharts, schematics), one #let fig-... per figure
├── figures/             make_*.py scripts and the SVG files they write
├── assets/logo.png      university logo, supplied by the author
└── build/thesis.pdf     output (git-ignored)
```

First run: copy `template/uet-thesis.typ` and `template/main.typ` from this skill into `thesis/src/`. Change layout only in `thesis/src/uet-thesis.typ`, and only to follow a rule in `format-rules.md`.

## Workflow

1. Read the skeleton and list which sections you will write and their status. Work on the sections asked for; do not rewrite others.
2. For each section, open its evidence files and draft the text in `thesis/src/chapters/`.
3. Tables: build them from the result files, with counts as well as percentages where the skeleton gives both. Caption above.
4. Figures: one script per figure in `thesis/figures/`, reading the result files named in the skeleton's "Figures and tables plan" and writing SVG. Load the `dataviz` skill before writing chart code. The thesis is laser-printed, so every chart must also read in greyscale: direct labels, distinct markers or line styles, no meaning carried by colour alone. Caption below.
5. Diagrams of mechanisms (inference flow, bridge steps, caches, adapter placement, pipelines) go in `thesis/src/diagrams.typ`, drawn with CeTZ (`@preview/cetz:0.4.2`): black strokes, white and grey fills, dashed outlines for optional or held-back parts. Inside CeTZ helpers call `draw.rect`/`draw.content` explicitly (`import draw: *` shadows a `fill` parameter), and set `par(first-line-indent: 0pt, justify: false)` for box text. Example tokens in a diagram are made up, so the caption says "illustrative". Each method or background section that describes a process should have one.
6. Mention every figure, table and appendix in the text before it appears, by reference (`@fig:name`, `@tab:name`, `@eq:name`), never by a typed number.
7. Build and check (next section). Fix every `FAIL` and every margin warning.
8. Look at the PDF once: render the pages you changed to PNG and read them for overflow, orphaned headings, and captions on the wrong side.
9. Update `thesis/pending.md` with every gap, then report: what was written, which sections remain `BLANK` or `PARTIAL`, the `[TO FILL]` markers from the build, and the open format points that affect this draft.

## Build and check

```powershell
uv run --no-project --with typst --with pymupdf python .claude/skills/thesis-pdf/scripts/build.py
```

It compiles `thesis/src/main.typ` to `thesis/build/thesis.pdf` (Typst root `thesis/`) and prints: page counts, body pages before the references against the 50 to 75 page rule, body font and size, line pitch, text outside the margins, abstract word count, and every `[TO FILL]`, `[TO CONFIRM]`, `[TO ADD]` and `[PENDING]` marker with its page. Exit code 1 means it did not compile or a hard rule failed.

To view pages:

```powershell
uv run --no-project --with pymupdf python -c "import pymupdf; d = pymupdf.open('thesis/build/thesis.pdf'); [d[i].get_pixmap(dpi=80).save(f'thesis/build/p{i + 1}.png') for i in (0, 11)]"
```

The line-pitch check needs a page of running text; on a near-empty draft it can warn without cause.

## What the template already does

Checked on 2026-11-24 with `typst` 0.15.2 (Python package) and placeholder text:

- A4; margins top 2.5 cm, bottom 3 cm, left 3 cm, right 2 cm; Times New Roman 13 pt; 19.4 pt line pitch; 6 pt between paragraphs; 1 cm first-line indent; justified.
- Framed cover, title page with supervisor lines, Authorship and Supervisor's approval with the required wording, Acknowledgement, optional Preface, Abstract at 12 pt with a keywords line.
- Table of contents (three levels), lists of figures, tables and algorithms, sorted abbreviations.
- `Chapter N` plus an upper-case title on a new page; sections `1.1.`, `1.1.1.`.
- Figures, tables, algorithms and equations numbered per chapter (`Figure 2.1 –`, `Table 2.1 –`, `(2.1)`); table and algorithm captions above, figure captions below; `A.1` numbering after `#show: appendix`.
- Roman page numbers in the front matter (hidden on the cover and title page), Arabic from Chapter 1 through the appendices, centred at the bottom.
- `#algorithm(caption: [...])[...]` for algorithm listings. (`#pending[...]` exists in the template but must not be used; gaps go to `thesis/pending.md`.)
- IEEE references through `#bibliography("refs.bib", title: [References], style: "ieee")`, placed before `#show: appendix`.

Not done by the template: the logo (pass `logo: image("../assets/logo.png", width: 2.6cm)` once the author supplies the file), landscape pages, and sub-figure grids (use a `grid` inside one `figure`, label parts (a), (b)).

## Done means

- The build exits 0 with no margin warning.
- Every number in the PDF traces to an evidence file named in the skeleton.
- The build lists no `[PENDING]` marker; front-matter placeholders (`[TO FILL]`, `[TO CONFIRM]`) are the only markers left in the PDF.
- Every gap is listed in `thesis/pending.md`.
- No file path appears in the PDF (search the extracted text for `/`, `.py`, `.json`, `.toml`, `.md`).
- No uncited external claim and no reference without a checked source.
