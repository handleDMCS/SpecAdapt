# Thesis format rules

Synthesised from the three references in `thesis/sample/`:

- **VN**: "Quy định về trình bày đồ án tốt nghiệp" (University of Engineering and Technology, VNU Hanoi), 7 pages. The detailed rulebook. Link in `thesis/sample/format.md`.
- **EN**: "Quy định về trình bày khóa luận tốt nghiệp (dành cho sinh viên chương trình đào tạo đạt chuẩn quốc tế)", 23 pages. The English template with cover pages and guidance text. Link in `thesis/sample/format.md`.
- **Sample**: `thesis/sample/21020007 (1).pdf`, an accepted 2024 Vietnamese thesis (86 pages, typeset with Typst). Shows how the rules are applied in practice.

The thesis is written in **English**. Use EN for structure and wording of the fixed pages, VN for every measurable layout rule, and the sample to settle what VN and EN leave open.

## 1. Page and type

| Rule | Value | Source |
| --- | --- | --- |
| Paper | A4 (210 × 297 mm), printed single-sided | VN §1 |
| Margins | top 2.5 cm, bottom 3 cm, left 3 cm, right 2 cm | VN §5 |
| Body font size | 13 pt | VN §5 |
| Font | A Unicode font. Times New Roman is named for the abstract and used throughout EN. Other typesetting systems (LaTeX and similar) are allowed; the sample uses New Computer Modern. | VN §5, §6, App. 03; EN; sample |
| Line spacing | 1.3 lines (about 19.4 pt baseline to baseline at 13 pt) | VN §5; measured 19.3 pt in the sample |
| Paragraphs | 6 pt between paragraphs, first line indented 1 cm, justified | VN §5; sample |
| Page numbers | centred at the bottom | VN §2 |
| Page numbering | Arabic, continuous from the Introduction to the end of the appendix. Front matter in lower-case Roman (EN shows v, vii, viii…; the sample shows i, ii, iii…). | VN §2; EN; sample |
| Length | at least about 50 to 75 pages, not counting tables, figures, references and appendices | VN §5 (the sample has 54 numbered pages before the references) |

## 2. Order of parts

Use this order (EN, matched by the sample):

1. Cover
2. Title page (adds supervisor and co-supervisor)
3. Authorship declaration
4. Supervisor's approval
5. Acknowledgement
6. Abstract (with keywords)
7. Table of contents
8. List of figures
9. List of tables
10. List of algorithms (only if the thesis has algorithm listings; the sample has one)
11. Abbreviations (only if any; alphabetical; do not abbreviate long or rarely used phrases)
12. Chapters, in the structure of section 4
13. Related publications (only if the author has any; the sample has this page)
14. References
15. Appendices

VN lists the abstract before the declaration and the appendix before the references. EN and the sample put references before appendices. Follow EN; see "Open points".

## 3. Fixed pages

**Cover** (EN App. 01): a double-line frame around the page; all lines centred and bold.

| Line | Style |
| --- | --- |
| `VIETNAM NATIONAL UNIVERSITY, HANOI` / `UNIVERSITY OF ENGINEERING AND TECHNOLOGY` | upper case, 12 pt |
| University logo | below the university lines |
| Author's name | lower case, 14 pt |
| THESIS TITLE | upper case, 18 pt |
| Thesis type line. VN: `ĐỒ ÁN TỐT NGHIỆP ĐẠI HỌC HỆ CHÍNH QUY`, upper case, 14 pt. EN's cover has no such line. | see "Open points" |
| `Major: <exact programme name>` | lower case, 14 pt |
| `HA NOI - 20<last two digits of the defense year>` | upper case, 12 pt |

**Title page** (EN App. 02): the cover again, without the frame, plus `Supervisor:` and `Co-Supervisor:` (lower case, bold, 14 pt) with room for signatures.

**Authorship** (EN), verbatim:

> "I hereby declare that the work contained in this thesis is of my own and has not been previously submitted for a degree or diploma at this or any other higher education institution. To the best of my knowledge and belief, the thesis contains no materials previously published or written by another person except where due reference or acknowledgement is made."
>
> Signature: ………

**Supervisor's approval** (EN), verbatim except the degree name, which the author must set:

> "I hereby approve that the thesis in its current form is ready for committee examination as a requirement for the Bachelor of Computer Science degree at the University of Engineering and Technology."
>
> Signature: ………

**Acknowledgement** (EN): supervisors, tutor, colleagues and friends, the department or lab, the university, any funding ("This thesis was partly supported by … under the project …"). The author writes this; do not invent names.

**Abstract** (EN + VN App. 03):

- Heading `ABSTRACT`, upper case, bold, centred.
- About 360 words, 12 pt (VN). One page, two at most (EN).
- Keywords: label in bold italic, 12 pt, at most 4 keyword phrases (VN).
- Cover these seven points in order (EN): context; the problem; motivation and idea; methodology; the methods used with their main parameters; results (at most the 3 most important); conclusions and their impact on the problem and the wider context.

**Preface** (EN, optional): the only place for personal remarks such as tasks that could not be finished. Goes before the abstract.

## 4. Headings and numbering

- Chapters, sections and subsections use Arabic numerals, never Roman: `Chapter 1`, `1.1.`, `1.1.1.` (VN §3). EN also shows a fourth level `1.1.1.1.`; the sample uses four levels sparingly. Prefer three.
- A chapter opens on a new page with `Chapter N` and then the title in upper case (EN).
- Chapter structure: the standard English thesis structure, chosen by the author on 2026-11-24.
  1. Introduction
  2. Literature review (a full chapter)
  3. Methodology (a full chapter)
  4. Results, in one or more chapters (this thesis uses two)
  5. Discussion
  6. Conclusion and future work
- EN's own chapter names are INTRODUCTION (Motivation; Contributions and thesis overview), RELATED WORK, THE METHOD, RESULTS AND DISCUSSIONS, CONCLUSIONS (Conclusions; Future Works). The chosen structure keeps that order, renames three chapters and separates results from discussion. The sample also renames and adds chapters, so EN's names are a default, not a requirement.
- VN requires the introduction to state why the topic is needed, its scientific and practical significance, the object and method of study and the content; and the conclusion to state the main results and assess them.

## 5. Figures, tables, equations, algorithms

- Numbered per chapter, separately per kind: `Figure 2.1`, `Table 2.1`, `(2.10)` (VN §4; EN).
- **Table captions go above the table. Figure captions go below the figure** (VN §4; EN).
- Caption form: `Figure 2.1 – Caption text.` EN prints figures as `Figure 2-1 –` and tables as `Table 2.1 –`; the sample uses `Hình 2.3 — …` with a bold label. Use the dotted form for both, bold label, en dash.
- Sub-figures are labelled (a), (b) and explained in one caption (EN).
- Landscape tables or figures: their top is the page's left edge (VN §4).
- Equations are numbered at the right margin, `(chapter.number)` (EN).
- Every figure, table and appendix is mentioned in the text, and appears after its first mention (EN).
- Refer to them by cross-reference, never by typed numbers (EN).
- Anything taken from another source (figure, table, quotation) carries a citation (VN §1; EN).
- Appendix items are lettered: `Figure A.1`, `Table C.1` (sample).

## 6. Writing rules

- Neutral, technical tone from the abstract to the references. Avoid "I", "we", "you" (EN).
- The main text is a complete technical document: a stated problem and its solution, nothing else (EN).
- Methods are described so that a peer could repeat the work: equipment, procedure in order, what was measured, sources of error (EN).
- Concise and clean; no corrections by hand (VN §1).
- Detailed derivations and supplementary results go to appendices (EN).

## 7. References

- Cited in the text by number in square brackets: `[12]` (VN App. 04 §7; EN).
- Only works actually read and cited or used (VN §2).
- Each entry: number in brackets, authors, title, source (journal, volume, issue, year or publisher), pages (VN App. 04 §3).
- Books and theses: title in italics. Articles: title in double quotes, journal or proceedings name in italics, then volume, year, `pp.` and pages (VN App. 04 §5, §6).
- EN gives IEEE-format examples, for instance:
  - `[9] H. Schimanski and C. Thanner, "Raiders of the lost ark," IEEE Trans. Electromagn. Compat., vol. 51, no. 5, pp. 543–547, May 2003.`
  - `[10] J. Matula and R. Franck, "A case for two," in Proc. 15th Int. Zurich Symp. …, Zurich, Switzerland, Feb. 2003, vol. 1, pp. 347–350.`
  - `[11] F.T. Ulaby, Fundamentals of Applied Electromagnetics, 2nd edn., Prentice Hall, 2004`
- Use IEEE style, numbered in order of first citation (EN; the sample also numbers by first citation).
- Foreign-language titles stay in the original language (VN App. 04 §2).

## 8. Printing and submission

- One hard-bound copy with gold lettering for the faculty, three soft-bound copies (committee, supervisor, reviewer) (VN §7).
- Laser printed, single-sided (VN §1).

## Open points to confirm with the supervisor

The references disagree or are silent here. The template follows the first option; say so when delivering a PDF.

| Point | Chosen | Alternative |
| --- | --- | --- |
| Chapter names | standard English thesis structure: Introduction, Literature review, Methodology, Results, Discussion, Conclusion and future work | EN's five names, with results and discussion in one chapter |
| Reference order | IEEE, by first citation (EN, sample) | grouped by language, alphabetical by author (VN App. 04 §2, §4) |
| References vs appendix | references first (EN, sample) | appendix first (VN §2) |
| Thesis type line on the cover | omitted, as on EN's cover | an English equivalent of VN's "ĐỒ ÁN TỐT NGHIỆP ĐẠI HỌC HỆ CHÍNH QUY" (the sample prints "KHÓA LUẬN TỐT NGHIỆP ĐẠI HỌC HỆ CHÍNH QUY") |
| Font | Times New Roman 13 pt | New Computer Modern 13 pt, as in the sample |
| Figure label form | `Figure 2.1 –` | `Figure 2-1 –` as printed in EN |
| Front-matter page numbers | Roman, counted from the cover, hidden on the cover and title page (EN) | Roman starting at the declaration (sample) |
| Degree name in the approval text | to be set by the author | EN's "Bachelor of Computer Science" |
| Abstract length | about 360 words (VN) | up to 2 pages (EN) |
