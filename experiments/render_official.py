"""Render report/content.md into the OFFICIAL Apart template (docx) and export a PDF with Pages.

Numbers come from experiments/build_report.py:numbers() (all computed from results/summary/*.csv).
Markup in content.md (deliberately small):
  % TITLE: ...            % AUTHOR: Name | Affiliation        % ABSTRACT ... % END
  ## Heading / ### Subheading
  paragraphs with **bold**, *italic*, `code`, ^sup^ and {{placeholders}}
  "1. " numbered and "- " bulleted lines
  ![Figure caption with **bold** lead](figures/file.png){6.3}   (width in inches)
  | table | rows |  (first row = header; separator row of dashes is skipped)
  :::box Title  ...lines...  :::
  <<pagebreak>>

    uv run python experiments/render_official.py
Writes report/official/Apart_report.docx and report/Apart_report.pdf; prints the page count, the
abstract word count and any unfilled placeholder.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import build_report  # same folder
import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "report"
TEMPLATE = REP / "official" / "apart_template_original.docx"
OUT_DOCX = REP / "official" / "Apart_report.docx"
OUT_PDF = REP / "Apart_report.pdf"
BODY_PT = 10
INLINE = re.compile(r"(\*\*.+?\*\*|\*.+?\*|`.+?`|\^.+?\^|~[^~\s]+?~)")


# ---------------------------------------------------------------- inline text -----------
def add_runs(par, text: str, size: float | None = None, italic_all: bool = False,
             color: RGBColor | None = None):
    for part in INLINE.split(text):
        if not part:
            continue
        bold = italic = code = sup = sub = False
        if part.startswith("**") and part.endswith("**"):
            part, bold = part[2:-2], True
        elif part.startswith("*") and part.endswith("*") and len(part) > 1:
            part, italic = part[1:-1], True
        elif part.startswith("`") and part.endswith("`"):
            part, code = part[1:-1], True
        elif part.startswith("^") and part.endswith("^"):
            part, sup = part[1:-1], True
        elif part.startswith("~") and part.endswith("~") and len(part) > 2:
            part, sub = part[1:-1], True
        run = par.add_run(part)
        run.bold = bold or None
        run.italic = (italic or italic_all) or None
        if code:
            run.font.name = "Courier New"
        if sup:
            run.font.superscript = True
        if sub:
            run.font.subscript = True
        if size:
            run.font.size = Pt(size)
        if color is not None:
            run.font.color.rgb = color
    return par


def fill(text: str, v: dict, missing: list) -> str:
    def sub(m):
        if m.group(1) in v:
            return str(v[m.group(1)])
        missing.append(m.group(1))
        return f"[{m.group(1)}]"

    return re.sub(r"\{\{([^{}]+)\}\}", sub, text)


# ---------------------------------------------------------------- template surgery ------
def set_cell_text_runs(paragraph, text):
    """Replace a paragraph's text while keeping its first run's formatting."""
    runs = paragraph.runs
    if not runs:
        paragraph.add_run(text)
        return
    runs[0].text = text
    for r in runs[1:]:
        if r._element.find(qn("w:footnoteReference")) is None:
            r.text = ""


def fill_front_matter(d, title: str, authors: list[tuple[str, str]], abstract: str):
    t0 = d.tables[0]
    title_par = next(p for p in t0.cell(0, 0).paragraphs if "PROJECT TITLE" in p.text)
    set_cell_text_runs(title_par, title)
    # author slots live in nested tables: "Author name k" / "Affiliation" paragraphs
    slots = []
    for p in t0._element.iter(qn("w:p")):
        txt = "".join(t.text or "" for t in p.iter(qn("w:t")))
        if txt.startswith("Author name") or txt.strip() == "Affiliation":
            slots.append(p)
    pairs = [(slots[i], slots[i + 1]) for i in range(0, len(slots) - 1, 2)
             if "".join(t.text or "" for t in slots[i].iter(qn("w:t"))).startswith("Author name")]
    for k, (pn, pa) in enumerate(pairs):
        if k >= len(authors):  # drop unused author slots entirely (they leave blank lines)
            for p in (pn, pa):
                p.getparent().remove(p)
            continue
        name, aff = authors[k]
        for p, txt in ((pn, name), (pa, aff)):
            ts = list(p.iter(qn("w:t")))
            if ts:
                ts[0].text = txt
                for t in ts[1:]:
                    t.text = ""
    # abstract: nested 1x1 table whose text starts with "Summarize your project"
    for tbl in t0._element.iter(qn("w:tbl")):
        txt = "".join(t.text or "" for t in tbl.iter(qn("w:t")))
        if txt.startswith("Summarize your project"):
            cell = tbl.find(qn("w:tr")).find(qn("w:tc"))
            pars = cell.findall(qn("w:p"))
            for p in pars[1:]:
                cell.remove(p)
            p = docx.text.paragraph.Paragraph(pars[0], d)
            for r in list(p.runs):
                r._element.getparent().remove(r._element)
            add_runs(p, abstract, size=BODY_PT, italic_all=True)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            break


def strip_body(d):
    """Remove the guidance info box and every body element after it (keep sectPr)."""
    body = d.element.body
    children = list(body.iterchildren())
    info = d.tables[1]._element
    start = children.index(info)
    for el in children[start:]:
        if el.tag != qn("w:sectPr"):
            body.remove(el)


# ---------------------------------------------------------------- body blocks -----------
def para(d, style="normal", align=None, space_after=4):
    p = d.add_paragraph(style=style)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(0)
    if align:
        p.alignment = align
    return p


def add_heading(d, text, level):
    p = d.add_paragraph(style="Heading 2" if level == 2 else "Heading 3")
    add_runs(p, text)
    p.paragraph_format.space_before = Pt(10 if level == 2 else 6)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True


def add_body(d, text):
    p = para(d, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_runs(p, text, size=BODY_PT)


def add_list_item(d, marker, text, size=BODY_PT, level=1):
    p = para(d, space_after=2)
    indent = 0.28 * level
    p.paragraph_format.left_indent = Inches(indent)
    p.paragraph_format.first_line_indent = Inches(-0.2)
    add_runs(p, f"{marker}\t" if marker != "•" else "•\t", size=size)
    add_runs(p, text, size=size)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(indent))


def add_figure(d, path, caption, width):
    p = para(d, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(REP / path), width=Inches(width))
    c = para(d, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=8)
    add_runs(c, caption, size=8.5, color=RGBColor(0x40, 0x40, 0x40))


def borders(table, color="BBBBBB"):
    tbl = table._element
    pr = tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "bottom", "insideH"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), "4")
        e.set(qn("w:color"), color)
        b.append(e)
    pr.append(b)


def add_table(d, rows):
    t = d.add_table(rows=len(rows), cols=len(rows[0]))
    borders(t)
    for i, r in enumerate(rows):
        for j, cell_text in enumerate(r):
            cell = t.cell(i, j)
            p = cell.paragraphs[0]
            add_runs(p, f"**{cell_text}**" if i == 0 else cell_text, size=8)
            p.paragraph_format.space_after = Pt(1)
            # keep the table on one page: every row stays with the next one
            p.paragraph_format.keep_with_next = i < len(rows) - 1
        trpr = t.rows[i]._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        trpr.append(cant)
    para(d, space_after=4)


def add_box(d, title, lines):
    t = d.add_table(rows=1, cols=1)
    cell = t.cell(0, 0)
    tcpr = cell._element.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), "F3F1EA")
    tcpr.append(shd)
    borders(t, "999999")
    p = cell.paragraphs[0]
    add_runs(p, f"**{title}**", size=9)
    for ln in lines:
        q = cell.add_paragraph()
        q.paragraph_format.space_after = Pt(1)
        q.paragraph_format.left_indent = Inches(0.15)
        q.paragraph_format.first_line_indent = Inches(-0.12)
        add_runs(q, "• " + ln.lstrip("- ").strip(), size=8.5)
    para(d, space_after=4)


def page_break(d):
    from docx.enum.text import WD_BREAK

    d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# ---------------------------------------------------------------- main ------------------
def render(text: str, v: dict):
    missing: list[str] = []
    text = fill(text, v, missing)
    lines = text.splitlines()
    title = next(ln[len("% TITLE:"):].strip() for ln in lines if ln.startswith("% TITLE:"))
    authors = [tuple(x.strip() for x in ln[len("% AUTHOR:"):].split("|"))
               for ln in lines if ln.startswith("% AUTHOR:")]
    a0 = lines.index("% ABSTRACT")
    a1 = lines.index("% END")
    abstract = " ".join(ln.strip() for ln in lines[a0 + 1:a1] if ln.strip())
    d = docx.Document(str(TEMPLATE))
    normal = d.styles["normal"]
    normal.font.size = Pt(BODY_PT)
    fill_front_matter(d, title, authors, abstract)
    strip_body(d)
    i, body = 0, lines[a1 + 1:]
    buf: list[str] = []

    def flush():
        if buf:
            add_body(d, " ".join(buf))
            buf.clear()

    while i < len(body):
        ln = body[i].rstrip()
        if not ln.strip():
            flush()
        elif ln.startswith("### "):
            flush()
            add_heading(d, ln[4:], 3)
        elif ln.startswith("## "):
            flush()
            add_heading(d, ln[3:], 2)
        elif ln.startswith("!["):
            flush()
            m = re.match(r"!\[(.*)\]\((.+?)\)(\{([\d.]+)\})?", ln)
            add_figure(d, m.group(2), m.group(1), float(m.group(4) or 6.3))
        elif ln.startswith("|"):
            flush()
            rows = []
            while i < len(body) and body[i].startswith("|"):
                cells = [c.strip() for c in body[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            add_table(d, rows)
            continue
        elif ln.startswith(":::box"):
            flush()
            title_b, items = ln[len(":::box"):].strip(), []
            i += 1
            while not body[i].startswith(":::"):
                if body[i].strip():
                    items.append(body[i])
                i += 1
            add_box(d, title_b, items)
        elif ln.strip() == "<<pagebreak>>":
            flush()
            page_break(d)
        elif re.match(r"^\d+\. ", ln):
            flush()
            k, rest = ln.split(". ", 1)
            add_list_item(d, f"{k}.", rest)
        elif ln.startswith("- "):
            flush()
            add_list_item(d, "•", ln[2:])
        elif ln.startswith("  - "):
            flush()
            add_list_item(d, "–", ln[4:], level=2)
        elif ln.startswith("% "):
            pass
        else:
            buf.append(ln.strip())
        i += 1
    flush()
    return d, abstract, missing


def tighten_front_matter(d):
    """Drop the empty spacer paragraphs between the template's title block and the body text."""
    t0 = d.tables[0]._tbl
    nxt = t0.getnext()
    if nxt is not None and nxt.tag == qn("w:p") and not "".join(nxt.itertext()).strip():
        nxt.getparent().remove(nxt)
    cell = d.tables[0].rows[-1].cells[0]
    last = cell.paragraphs[-1]
    if not last.text.strip():
        last.paragraph_format.space_before = Pt(0)
        last.paragraph_format.space_after = Pt(0)
        last.paragraph_format.line_spacing = Pt(1)


def main():
    v = build_report.numbers()
    d, abstract, missing = render((REP / "content.md").read_text(), v)
    tighten_front_matter(d)
    OUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    d.save(str(OUT_DOCX))
    strict = len([w for w in re.split(r"[\s\-–—/]+", abstract) if re.search(r"\w", w)])
    print(f"abstract words: {len(abstract.split())} (strict, hyphens and dashes split: {strict})")
    # plain-text abstract for the submission form's "Project summary" field
    plain = re.sub(r"[*`~^]", "", abstract)
    (REP / "abstract_for_form.txt").write_text(plain + "\n")
    script = f'''with timeout of 180 seconds
tell application "Pages"
  set doc to open POSIX file "{OUT_DOCX}"
  export doc to POSIX file "{OUT_PDF}" as PDF
  close doc saving no
end tell
end timeout'''
    for attempt in range(3):  # Pages occasionally fails transiently
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=False)
        if r.returncode == 0:
            break
        print(f"Pages export attempt {attempt + 1} failed: {r.stderr.strip()[:200]}")
    else:
        raise RuntimeError("Pages export failed three times")
    info = subprocess.run(["pdfinfo", str(OUT_PDF)], capture_output=True, text=True, check=False)
    pages = re.search(r"Pages:\s+(\d+)", info.stdout)
    print(f"pages: {pages.group(1) if pages else '?'}  -> {OUT_PDF}")
    print(f"unfilled placeholders: {sorted(set(missing))}")


if __name__ == "__main__":
    main()
