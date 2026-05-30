# WEEKLY_REPORT_GUIDELINE.md

Formatting spec for the weekly progress report, reverse-engineered from
`Technical_Weekly_Report_11.docx`. It is a Google Docs export, so the
structure maps cleanly to Markdown heading levels. When generating a new
weekly report, follow this exactly so the converted `.docx` matches the
existing house style.

The canonical companion file for academic-style reports lives at
`RESEARCH_REPORT_RULES.md`; this file covers the weekly progress format
only.

---

## 1. Heading hierarchy & colors

Every heading uses the same blue (`#4F81BD`), Calibri font, **not bold**.
Size is the only thing that distinguishes levels.

| Element | Markdown | Style | Font | Size | Color | Example |
|---|---|---|---|---|---|---|
| Document title | `#` | Heading 1 | Calibri | 16pt | `#4F81BD` | Weekly Progress Report |
| Section | `###` | Heading 3 | Calibri | 12pt | `#4F81BD` | `1. Region-Based Stop-Motion Web App` |
| Subsection | `####` | Heading 4 | Calibri | 12pt | `#4F81BD` | `Role in Pipeline`, `Motivation`, `Workflow` |
| Body text | (plain) | Normal | Cambria | 12pt | black | paragraphs |
| *(unused)* Title | — | Title | Calibri | 18pt | `#335B8A` | — |
| *(unused)* Subtitle | — | Subtitle | Calibri | 15pt | `#335B8A` | — |

Note: Heading 3 and Heading 4 are the *same* size (12pt) — they're only
differentiated semantically, not visually, except that H4 subsections
tend to read as bold because Calibri sits next to the Cambria body
text. The date line (`Work Completed (This Week) – 21th April, 2026`)
is also Heading 3.

---

## 2. Document structure pattern

```
# Weekly Progress Report
### Work Completed (This Week) – <date>
### 1. <Section Name>
#### <Subsection>          ← Role in Pipeline / Motivation / Workflow / etc.
<body paragraph>
- <bullet>
#### <Subsection>
...
### 2. <Section Name>
...
```

Recurring subsection labels used in prior reports:

- *Role in Pipeline*
- *Workflow*
- *Backend Pipeline*
- *Motivation*
- *Approach*
- *Results*
- *Conclusion*
- *Current Status and Open Questions*
- *Stage-wise Progress Overview*

---

## 3. Other formatting

- **Page**: US Letter (12240 × 15840 DXA), 1-inch margins all sides.
- **Lists**: real bullet lists (not `•` characters). Lead-in terms
  bolded inline, e.g. **Select Video**: description.
- **Tables**: full single black borders (`sz=12`), no fill/shading on
  any cell. Header row is plain (just `tblHeader=1` so it repeats
  across pages), bold header text. No colored cells anywhere.
- **Inline color**: body runs are black only; no colored body text.

---

## 4. Reusable LLM prompt block

Paste this into any model to generate a report in this exact format:

```
Write a Weekly Progress Report in Markdown with this structure:
- H1 (#): "Weekly Progress Report"
- H3 (###): "Work Completed (This Week) – <date>"
- H3 (###): numbered sections "1. <Title>", "2. <Title>", ...
- H4 (####) subsections inside each section, chosen from:
  Motivation, Role in Pipeline, Workflow, Approach,
  Results, Conclusion, Current Status and Open Questions
- Body text in plain paragraphs (concise, technical, factual).
- Bullet lists with the lead-in term in **bold**, then a colon and explanation.
- Use Markdown tables for stage/status summaries; no colored cells.
Keep tone factual and concise, no filler.
```

When you convert the resulting Markdown to `.docx`, apply:
- all headings = Calibri, color `#4F81BD`, not bold
- body = Cambria 12pt
- tables with single black borders and no shading

That reproduces the existing template exactly.
