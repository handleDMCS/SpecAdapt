---
name: "first-principle-summa"
description: "Summarize a technical process using first principles, broken into steps with tl;dr, why, input, and output."
---

Summarize a technical set-up/process using first principles in the following format; put this at the very top of your response : 

**Codex only:** write the response as a Markdown file in `./first_principle_notes/` (from the root dir), named after the topic in kebab-case (e.g. `my-topic.md`); create the folder first if it doesn't exist.

**Overview**
tl;dr: a brief overview of the process 1-3 lines. This is the bigger picture


Then do this for the follow-up: 4-5 lines/step; make sure each step is self-contained and does exactly one thing.


**{step name}**
**tl;dr**: 1-liner; avoid technical language; just give a literal description of what happens in this step.

**Why**: where does it fit in the bigger picture?

**Input**: what data/resources does this step consume? Avoid technical language; format as a bulleted list in a code tab, 1 line/item; avoid technical language;

**Output**: what data/resources does this step output? Avoid technical language; format as a bulleted list in a code tab, 1 line/item; avoid technical language;

**Technical keywords**: a list of technical keywords associated with this step, for referencing and research purposes. This is optional, 1-5 keywords, contained within 1 line.

Check :

- The final explanation should be concise (zero fluff), packed with information, and friendly to both technical and non-technical people.

- Visually smooth: only "Input" and "Output" have code tabs attached to them. 

- The technical keywords are actually necessary if they're included.