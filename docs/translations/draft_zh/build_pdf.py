"""Build the Chinese Draft translation using ReportLab and local fonts.

Run from any directory with system Python 3; no network access is required.
"""

from pathlib import Path
import html
import re

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "Draft_中文全文译本.pdf"
pdfmetrics.registerFont(TTFont("CJK", "/usr/share/fonts/truetype/arphic/uming.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("Latin", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("LatinBold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
pdfmetrics.registerFontFamily("CJK", normal="CJK", bold="CJK", italic="CJK", boldItalic="CJK")
pdfmetrics.registerFontFamily("Latin", normal="Latin", bold="LatinBold", italic="Latin", boldItalic="LatinBold")

INK = colors.HexColor("#172b3a")
BLUE = colors.HexColor("#245879")
GRAY = colors.HexColor("#566773")
PAGE_W, PAGE_H = A4
WIDTH = PAGE_W - 100
styles = {
    "body": ParagraphStyle("body", fontName="CJK", fontSize=10.5, leading=17.5,
                           textColor=INK, wordWrap="CJK", spaceAfter=8),
}
styles["title"] = ParagraphStyle("title", parent=styles["body"], fontSize=24, leading=35,
                                  textColor=BLUE, spaceAfter=24)
styles["meta"] = ParagraphStyle("meta", parent=styles["body"], fontSize=10.5, leading=18,
                                 textColor=GRAY, spaceAfter=12)
styles["note"] = ParagraphStyle("note", parent=styles["body"], fontSize=9, leading=15,
                                 textColor=GRAY, backColor=colors.HexColor("#f2f5f7"),
                                 borderPadding=10, spaceBefore=14, spaceAfter=14)
for name, size, leading in [("h1", 16, 24), ("h2", 12.5, 20), ("h3", 11, 18)]:
    styles[name] = ParagraphStyle(name, parent=styles["body"], fontSize=size, leading=leading,
                                 textColor=BLUE, spaceBefore=15, spaceAfter=8, keepWithNext=True)
styles["caption"] = ParagraphStyle("caption", parent=styles["body"], fontSize=9, leading=14,
                                    textColor=GRAY, spaceAfter=7)
styles["legend"] = ParagraphStyle("legend", parent=styles["caption"], fontSize=8.5, leading=13)
styles["equation"] = ParagraphStyle("equation", parent=styles["body"], fontName="CJK",
                                     fontSize=11, leading=22, alignment=TA_CENTER,
                                     spaceBefore=8, spaceAfter=12)
styles["cell"] = ParagraphStyle("cell", parent=styles["body"], fontSize=8.4, leading=12,
                                 alignment=TA_CENTER, spaceAfter=0)
styles["ref"] = ParagraphStyle("ref", parent=styles["body"], fontSize=8.7, leading=14,
                                spaceAfter=9)


def markup(text):
    """Escape text while retaining the small, explicit inline markup vocabulary."""
    text = html.escape(text)
    for tag in ("sub", "sup"):
        text = text.replace(f"&lt;{tag}&gt;", f"<{tag}>")
        text = text.replace(f"&lt;/{tag}&gt;", f"</{tag}>")
    text = text.replace("<sup>", "<super>").replace("</sup>", "</super>")
    text = re.sub(r"\*\*(.+?)\*\*", r'<font name="LatinBold">\1</font>', text)
    # Render Latin and mathematical characters with an embedded Latin font;
    # keep Chinese text and punctuation in the embedded CJK font.
    pieces = re.split(r"(<[^>]+>)", text)
    in_bold = False
    rendered = []
    for piece in pieces:
        if piece.startswith("<"):
            if piece == '<font name="LatinBold">':
                in_bold = True
            elif piece == '</font>':
                in_bold = False
            rendered.append(piece)
        elif in_bold:
            rendered.append(piece)
        else:
            rendered.append(re.sub(r"([\x20-\u2e7f]+)", r'<font name="Latin">\1</font>', piece))
    return "".join(rendered)


class TranslationDoc(BaseDocTemplate):
    def __init__(self):
        super().__init__(str(OUTPUT), pagesize=A4, leftMargin=50, rightMargin=50,
                         topMargin=51, bottomMargin=47,
                         title="Draft：面向机器人设计探索的参数化工具｜中文全文译本",
                         author="David Nguyen; Marcelo Coelho; Sangbae Kim（原作者）",
                         subject="arXiv:2609.38405v1 中文全文翻译",
                         pageCompression=1)
        self.addPageTemplates(PageTemplate(id="main", frames=[Frame(
            50, 47, WIDTH, PAGE_H - 98, leftPadding=0, rightPadding=0,
            topPadding=0, bottomPadding=0)], onPage=self.header_footer))

    def header_footer(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("CJK", 8)
        canvas.setFillColor(GRAY)
        if doc.page > 1:
            canvas.drawString(50, PAGE_H - 28, "Draft｜面向机器人设计探索的参数化工具")
            canvas.drawRightString(PAGE_W - 50, PAGE_H - 28, "中文全文译本")
            canvas.setStrokeColor(colors.HexColor("#d6e0e6"))
            canvas.line(50, PAGE_H - 36, PAGE_W - 50, PAGE_H - 36)
        canvas.drawString(50, 25, "原文：arXiv:2609.38405v1　｜　非官方译本")
        canvas.drawRightString(PAGE_W - 50, 25, str(doc.page))
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and hasattr(flowable, "outline_key"):
            level = flowable.outline_level
            key = flowable.outline_key
            label = flowable.getPlainText()
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(label, key, level=level, closed=False)
            self.notify("TOCEntry", (level, label, self.page, key))


def make_table(lines):
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    rows = [row for row in rows if not all(re.fullmatch(r":?-+:?", c) for c in row)]
    n = len(rows[0])
    if n == 5:
        widths = [100, 210, 44, 55, WIDTH - 409]
    elif n == 3:
        widths = [125, 105, WIDTH - 230]
    elif n == 6:
        widths = [100, 75, 65, 80, 80, WIDTH - 400]
    elif n == 9:
        widths = [72] + [(WIDTH - 72) / 8] * 8
    elif n == 10:
        widths = [62] + [(WIDTH - 62) / 9] * 9
    else:
        widths = [WIDTH / n] * n
    table = Table([[Paragraph(markup(c), styles["cell"]) for c in row] for row in rows],
                  colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f0f5")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fa")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, BLUE),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, BLUE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def build():
    blocks = re.split(r"\n\s*\n", (HERE / "translation.md").read_text().strip())
    story = []
    cover = True
    refs = False
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("toc0", fontName="CJK", fontSize=11, leading=20, spaceBefore=7,
                       textColor=BLUE),
        ParagraphStyle("toc1", fontName="CJK", fontSize=10, leading=17, leftIndent=16,
                       textColor=INK),
    ]
    heading_id = 0
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if b.startswith("# "):
            story.extend([Spacer(1, 66), Paragraph(markup(b[2:]), styles["title"]), Spacer(1, 10)])
        elif b.startswith("> "):
            story.append(Paragraph(markup(b[2:]), styles["note"]))
        elif b.startswith("## "):
            if cover:
                cover = False
                story.extend([PageBreak(), Paragraph("目录", styles["h1"]), Spacer(1, 10), toc, PageBreak()])
            if b[3:] == "参考文献":
                refs = True
                story.append(PageBreak())
            p = Paragraph(markup(b[3:]), styles["h1"])
            p.outline_key, p.outline_level = f"section-{heading_id}", 0
            heading_id += 1
            story.append(p)
        elif b.startswith("### "):
            p = Paragraph(markup(b[4:]), styles["h2"])
            p.outline_key, p.outline_level = f"section-{heading_id}", 1
            heading_id += 1
            story.append(p)
        elif b.startswith("#### "):
            story.append(Paragraph(markup(b[5:]), styles["h3"]))
        elif b.startswith("!["):
            path = HERE / re.search(r"\]\((.+)\)", b).group(1)
            w, h = PILImage.open(path).size
            fig_num = int(re.search(r"figure_(\d+)", str(path)).group(1))
            display_w = WIDTH if fig_num in (4, 5, 8) else WIDTH * (0.76 if fig_num in (1, 3, 7) else 0.88)
            display_h = display_w * h / w
            img = Image(str(path), width=display_w, height=display_h)
            img.hAlign = "CENTER"
            caption = blocks[i + 1]
            figure = [Spacer(1, 8), img, Spacer(1, 6), Paragraph(markup(caption), styles["caption"])]
            story.append(KeepTogether(figure))
            i += 1
        elif b.startswith("图内标签对照"):
            story.append(Paragraph(markup(b), styles["legend"]))
        elif re.match(r"^表 [IVX]+\u3000", b):
            caption = Paragraph(markup(b), styles["caption"])
            caption.keepWithNext = True
            story.extend([Spacer(1, 8), caption])
        elif b.startswith("|"):
            story.extend([make_table(b.splitlines()), Spacer(1, 12)])
        elif b.startswith("$$ "):
            story.append(Paragraph(markup(b[3:]), styles["equation"]))
        elif re.match(r"^\d+\. ", b):
            for item in b.splitlines():
                story.append(Paragraph(markup(item), styles["body"]))
        else:
            style = styles["meta"] if cover else styles["ref"] if refs else styles["body"]
            story.append(Paragraph(markup(b.replace("\n", " ")), style))
        i += 1
    # Keep short subsection headings with their first paragraph across TOC passes.
    grouped = []
    j = 0
    while j < len(story):
        flow = story[j]
        if (isinstance(flow, Paragraph) and flow.style.name == "h3"
                and j + 1 < len(story) and isinstance(story[j + 1], Paragraph)):
            grouped.append(KeepTogether([flow, story[j + 1]]))
            j += 2
        else:
            grouped.append(flow)
            j += 1
    doc = TranslationDoc()
    doc.multiBuild(grouped)
    print(f"PDF written: {OUTPUT}")
    print(f"Pages: {doc.page}; headings: {heading_id}")


if __name__ == "__main__":
    build()
