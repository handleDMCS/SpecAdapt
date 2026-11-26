// Layout for an English graduation thesis following the rules in ../format-rules.md.
// Copy this file next to main.typ; do not edit the copy in the skill folder.
//
//   #import "uet-thesis.typ": *
//   #show: thesis.with(title: [...], author: [...], ...)
//   = Introduction
//   ...
//   #bibliography("refs.bib", title: [References], style: "ieee")
//   #show: appendix
//   = Reproducibility

#let in-appendix = state("thesis-in-appendix", false)

// "2.1" in chapters, "A.1" in appendices; used for figures, tables, algorithms and equations.
#let chapter-item(pattern-main, pattern-appendix) = n => {
  let chapter = counter(heading).get().first()
  numbering(if in-appendix.get() { pattern-appendix } else { pattern-main }, chapter, n)
}

// A visible marker for content that is not available yet. build.py lists every one.
#let pending(body) = text(fill: rgb("#a3302a"), weight: "bold")[\[PENDING: #body\]]

#let algorithm(caption: none, body) = figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: caption,
  block(width: 100%, stroke: (top: 0.8pt, bottom: 0.8pt), inset: (y: 6pt), align(left, body)),
)

#let front-heading(body, outlined: false) = heading(numbering: none, outlined: outlined, body)

#let cover-page(framed: true, university: (), logo: none, author: [], title: [], thesis-type: none,
                major: [], supervisors: none, place-year: []) = page(numbering: none)[
  #set par(first-line-indent: 0pt, justify: false, leading: 0.6em, spacing: 0.9em)
  #set align(center)
  #let inner = {
    for line in university { text(size: 12pt, weight: "bold", upper(line)); linebreak() }
    if logo != none { v(0.8cm); logo }
    v(1fr)
    text(size: 14pt, weight: "bold", author)
    v(1fr)
    text(size: 18pt, weight: "bold", upper(title))
    v(1fr)
    if thesis-type != none { text(size: 14pt, weight: "bold", upper(thesis-type)); v(0.4cm) }
    text(size: 14pt, weight: "bold")[Major: #major]
    if supervisors != none {
      v(1fr)
      align(left, pad(left: 1.5cm, text(size: 14pt, weight: "bold", supervisors)))
    }
    v(2fr)
    text(size: 12pt, weight: "bold", upper(place-year))
  }
  #if framed {
    rect(width: 100%, height: 100%, stroke: 2.5pt, inset: 3pt,
      rect(width: 100%, height: 100%, stroke: 0.8pt, inset: 0.8cm, inner))
  } else { inner }
]

#let thesis(
  title: [\[TO FILL: thesis title\]],
  author: [\[TO FILL: author\]],
  major: [\[TO FILL: major\]],
  supervisor: [\[TO FILL: supervisor\]],
  co-supervisor: none,
  university: ("Vietnam National University, Hanoi", "University of Engineering and Technology"),
  place-year: [Ha Noi - 20\_\_],
  degree: [\[TO FILL: degree name\]],
  thesis-type: none,          // optional cover line, see "Open points" in format-rules.md
  logo: none,                 // content, for example image("../assets/logo.png", width: 2.6cm)
  preface: none,
  acknowledgement: [\[TO FILL: acknowledgement, written by the author\]],
  abstract: [\[TO FILL: abstract\]],
  keywords: (),
  abbreviations: (),          // array of (short, long) pairs
  list-of-algorithms: false,
  body-font: ("Times New Roman", "New Computer Modern"),
  body,
) = {
  set document(title: title, author: ())
  set page(paper: "a4", margin: (top: 2.5cm, bottom: 3cm, left: 3cm, right: 2cm),
           numbering: "i", number-align: center + bottom)
  set text(font: body-font, size: 13pt, lang: "en")
  // 1.3 lines at 13 pt is about 19.4 pt baseline to baseline; 6 pt more between paragraphs.
  set par(justify: true, leading: 0.83em, spacing: 0.83em + 6pt, first-line-indent: (amount: 1cm, all: true))
  show raw: set text(font: ("Consolas", "DejaVu Sans Mono"), size: 10.5pt)

  set heading(numbering: "1.1.")
  show heading: set par(first-line-indent: 0pt, justify: false)
  show heading.where(level: 1): it => {
    pagebreak(weak: true)
    counter(figure.where(kind: image)).update(0)
    counter(figure.where(kind: table)).update(0)
    counter(figure.where(kind: "algorithm")).update(0)
    counter(math.equation).update(0)
    if it.numbering == none {
      block(above: 0pt, below: 1cm, width: 100%, align(center, text(size: 16pt, weight: "bold", upper(it.body))))
    } else {
      block(above: 0pt, below: 1cm, {
        v(1cm)
        text(size: 16pt, weight: "bold", context {
          if in-appendix.get() [Appendix #counter(heading).display("A")] else [Chapter #counter(heading).display("1")]
        })
        v(0.5cm)
        text(size: 18pt, weight: "bold", upper(it.body))
      })
    }
  }
  show heading.where(level: 2): set text(size: 14pt)
  show heading.where(level: 2): set block(above: 1.4em, below: 0.9em)
  show heading.where(level: 3): set text(size: 13pt)
  show heading.where(level: 3): set block(above: 1.2em, below: 0.8em)
  show heading.where(level: 4): set text(size: 13pt, style: "italic")

  set figure(numbering: chapter-item("1.1", "A.1"))
  set figure.caption(separator: [ – ])
  show figure.caption: it => context {
    set par(first-line-indent: 0pt, justify: false)
    [*#it.supplement #it.counter.display(it.numbering)*#it.separator#it.body]
  }
  show figure.where(kind: table): set figure.caption(position: top)
  show figure.where(kind: "algorithm"): set figure.caption(position: top)
  show figure: set block(above: 1.3em, below: 1.3em)
  show table: set text(size: 12pt)
  show table: set par(justify: false, leading: 0.6em)
  set table(stroke: 0.5pt, inset: 6pt)
  set math.equation(numbering: chapter-item("(1.1)", "(A.1)"))

  // Fixed pages.
  cover-page(framed: true, university: university, logo: logo, author: author, title: title,
             thesis-type: thesis-type, major: major, place-year: place-year)
  cover-page(framed: false, university: university, logo: logo, author: author, title: title,
             thesis-type: thesis-type, major: major, place-year: place-year,
             supervisors: {
               [Supervisor: #supervisor]
               if co-supervisor != none { v(1.6cm); [Co-Supervisor: #co-supervisor] }
             })

  front-heading[Authorship]
  ["I hereby declare that the work contained in this thesis is of my own and has not been previously submitted for a degree or diploma at this or any other higher education institution. To the best of my knowledge and belief, the thesis contains no materials previously published or written by another person except where due reference or acknowledgement is made."]
  v(1.5cm)
  [Signature: #box(width: 7cm, repeat[.])]

  front-heading[Supervisor's approval]
  ["I hereby approve that the thesis in its current form is ready for committee examination as a requirement for the #degree degree at the University of Engineering and Technology."]
  v(1.5cm)
  [Signature: #box(width: 7cm, repeat[.])]

  front-heading[Acknowledgement]
  acknowledgement

  if preface != none {
    front-heading[Preface]
    preface
  }

  front-heading[Abstract]
  {
    set text(size: 12pt)
    abstract
    parbreak()
    text(weight: "bold", style: "italic")[Keywords: ]
    if keywords.len() == 0 [\[TO FILL: at most four keyword phrases\]] else { keywords.join(", ") }
  }

  {
    show outline.entry.where(level: 1): set text(weight: "bold")
    outline(title: [Table of contents], depth: 3)
  }
  {
    show heading: set heading(outlined: true)
    outline(title: [List of figures], target: figure.where(kind: image))
    outline(title: [List of tables], target: figure.where(kind: table))
    if list-of-algorithms { outline(title: [List of algorithms], target: figure.where(kind: "algorithm")) }
    if abbreviations.len() > 0 {
      front-heading(outlined: true)[Abbreviations]
      set par(first-line-indent: 0pt)
      grid(columns: (3.5cm, 1fr), row-gutter: 0.9em,
        ..abbreviations.sorted(key: pair => lower(pair.at(0))).map(pair => (strong(pair.at(0)), pair.at(1))).flatten())
    }
  }

  // Main matter: Arabic page numbers from the first chapter to the end of the appendices.
  set page(numbering: "1")
  counter(page).update(1)
  body
}

// Everything after `#show: appendix` is lettered: Appendix A, Figure A.1, Table A.1.
#let appendix(body) = {
  set heading(numbering: "A.1.")
  counter(heading).update(0)
  in-appendix.update(true)
  body
}
