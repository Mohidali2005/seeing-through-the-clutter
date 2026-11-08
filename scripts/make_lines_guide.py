#!/usr/bin/env python
"""
Build docs/guides/Lines-VQA_Teammate_Guide.pdf (same layout as the other teammate guides).

Needs the built dataset (data/lines_vqa/) for the example images and reportlab:
    python scripts/make_lines_guide.py [--data data/lines_vqa] [--scene 1]
                                       [--out docs/guides/Lines-VQA_Teammate_Guide.pdf]
"""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

from PIL import Image as PILImage, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ACCENT = colors.HexColor("#6b3fa0")
ACCENT_BG = colors.HexColor("#f3eefa")
GREY_BG = colors.HexColor("#f2f2f2")
LINE = colors.HexColor("#c8c8c8")
TEXT = colors.HexColor("#222222")
MUTED = colors.HexColor("#666666")
REPO = "atml-shapes-vqa/lines-vqa"
VERSIONS = ["clean", "crossings_0", "crossings_2", "crossings_4", "crossings_8", "crossings_4_colored"]

S = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=22, leading=27, textColor=TEXT),
    "sub": ParagraphStyle("sub", fontName="Helvetica", fontSize=9.5, leading=13, textColor=MUTED),
    "h": ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=ACCENT, spaceBefore=10, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.5, leading=13.5, textColor=TEXT, alignment=TA_LEFT),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8.5, leading=11, textColor=TEXT),
    "cellb": ParagraphStyle("cellb", fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=TEXT),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=7.8, leading=10, textColor=TEXT),
    "bullet": ParagraphStyle("bullet", fontName="Helvetica", fontSize=9.5, leading=13.5, textColor=TEXT, leftIndent=10,
                             firstLineIndent=-10, spaceAfter=3),
}


def P(text, style="body"):
    return Paragraph(text, S[style])


def mono(text):
    return f'<font name="Courier">{text}</font>'


def table(rows, widths, header=True):
    data = [[P(c, "cellb" if header and i == 0 else "cell") for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), GREY_BG), ("LINEBELOW", (0, 0), (-1, 0), 0.8, MUTED)]
    t.setStyle(TableStyle(style))
    return t


def code(text, width):
    t = Table([[Preformatted(text.strip("\n"), S["code"])]], colWidths=[width])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), GREY_BG), ("BOX", (0, 0), (-1, -1), 0.4, LINE),
                           ("LEFTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return t


def callout(text, width):
    t = Table([[P(text)]], colWidths=[width])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), ACCENT_BG), ("LINEBEFORE", (0, 0), (0, -1), 3, ACCENT),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 7),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return t


def example_strip(root, scene, width):
    """One scene in all 6 versions (2 rows x 3), with labels, as a reportlab Image."""
    rows = {r["version"]: r for r in csv.DictReader(open(root / "metadata.csv", encoding="utf-8"))
            if int(r["scene_id"]) == scene}
    tile, pad, head, foot = 360, 12, 30, 34
    W = pad + 3 * (tile + pad)
    H = pad + 2 * (head + tile + pad) + foot
    sheet = PILImage.new("RGB", (W, H), (236, 236, 236))
    d = ImageDraw.Draw(sheet)
    f_head, f_foot = ImageFont.load_default(size=19), ImageFont.load_default(size=19)
    for i, v in enumerate(VERSIONS):
        x, y = pad + (i % 3) * (tile + pad), pad + (i // 3) * (head + tile + pad)
        with PILImage.open(root / rows[v]["file_name"]) as im:
            sheet.paste(im.convert("RGB").resize((tile, tile), PILImage.LANCZOS), (x, y + head))
        d.text((x, y + 5), f"scene {scene}  |  {v}", fill=(0, 0, 0), font=f_head)
    r = rows["clean"]
    d.text((pad, H - foot + 6), f"follow the line from {r['target_dot']}  ->  answer (all versions): {r['answer']}"
           f"   [{r['split']}]", fill=(0, 0, 0), font=f_foot)
    buf = io.BytesIO()
    sheet.save(buf, format="PNG")
    buf.seek(0)
    img = Image(buf, width=width, height=width * H / W)
    img.hAlign = "CENTER"
    return img


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.2 * cm, "Lines-VQA - teammate guide")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"page {doc.page}")
    canvas.restoreState()


def build(root, scene, out):
    info_rows = list(csv.DictReader(open(root / "metadata.csv", encoding="utf-8")))
    question = next(r["question"] for r in info_rows if r["target_dot"] == "B")
    split_imgs = {s: sum(r["split"] == s for r in info_rows) for s in ("train", "validation", "test")}
    split_scn = {s: n // len(VERSIONS) for s, n in split_imgs.items()}

    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=2 * cm, title="Lines-VQA: How to Use It", author="Mohid Ali")
    w = A4[0] - 4 * cm
    st = []
    st += [P("Lines-VQA: How to Use It", "title"), Spacer(1, 2),
           P(f"ATML project&nbsp;&nbsp;|&nbsp;&nbsp;huggingface.co/datasets/{REPO}&nbsp;&nbsp;|&nbsp;&nbsp;questions: ask Mohid", "sub")]

    st += [P("What it is", "h"),
           P("Our fourth dataset, and a different skill: <b>following a line</b> instead of counting. Four dots "
             "labelled <b>A-D</b> sit on the left, eight numbered <b>boxes 1-8</b> on the right, and lines run from "
             "dots to boxes. The question names one dot; the answer is the box its line ends in. Every place where "
             "another line crosses the asked-about line (the <i>target</i>) is a spot where the two are tangled "
             "together, so the <b>number of crossings is a direct dial for entanglement</b>. "
             f"<b>500 scenes x 6 versions = {len(info_rows):,} images</b> (512 x 512)."),
           Spacer(1, 6),
           table([["Version", "Diff.", "What is drawn besides the target line", "What it tests"],
                  ["clean", "0", "nothing (the dots and boxes are always drawn)", "baseline"],
                  ["crossings_0", "1", "lines from the 3 other dots; <b>none</b> crosses the target", "plain clutter"],
                  ["crossings_2", "2", "the 3 lines cross the target <b>2</b> times in total", "entanglement"],
                  ["crossings_4", "3", "... <b>4</b> times", "entanglement"],
                  ["crossings_8", "4", "... <b>8</b> times", "entanglement"],
                  ["crossings_4_colored", "5", "the <b>same lines</b> as crossings_4, but coloured (target stays black)",
                   "does separating the target help?"]],
                 [3.4 * cm, 1.1 * cm, 7.6 * cm, 4.9 * cm]),
           Spacer(1, 8),
           callout("<b>Key idea:</b> the target line, the dots and the boxes are pixel-identical in all 6 versions "
                   "of a scene, so <b>all 6 versions have the same correct answer</b>. crossings_0 to crossings_8 "
                   "differ only in how often the other lines cross the target, so they give <b>accuracy vs. "
                   "entanglement</b>. crossings_4 and crossings_4_colored have exactly the same lines; only the "
                   "colour differs.", w),
           Spacer(1, 8),
           example_strip(root, scene, 0.84 * w),
           Spacer(1, 6),
           KeepTogether([P("The question is the same for every image except the letter:"),
                         Spacer(1, 3),
                         code(question.replace("Which numbered", "\nWhich numbered").replace("Lines may", "\nLines may")
                              .replace("Answer in", "\nAnswer in") + "\n\nanswer:  box: 5", w)]),
           Spacer(1, 5),
           P("Every crossing is a clean X (at least 45 degrees), lines never touch except where they cross, and the "
             "other lines never cross each other, so every crossing is on the target. Crossing counts are even "
             "because of geometry. The other lines may end in different boxes in different versions; the target "
             f"never changes. Splits are by scene: <b>train</b> {split_imgs['train']:,} images ({split_scn['train']} "
             f"scenes), <b>validation</b> {split_imgs['validation']} ({split_scn['validation']}), <b>test</b> "
             f"{split_imgs['test']} ({split_scn['test']}).")]

    st += [P("Useful columns", "h"),
           table([["Column", "Meaning"],
                  [mono("image_id"), "e.g. line_0042_crossings_4. The key for your predictions"],
                  [mono("scene_id"), "same for all 6 versions of a scene. Use it to compare versions"],
                  [mono("version, difficulty"), "clean / crossings_0 / _2 / _4 / _8 / crossings_4_colored = 0-5"],
                  [mono("target_dot, target_box"), "where the target line starts and ends (target_box is the answer)"],
                  [mono("num_crossings"), "how often other lines cross the target (0, 2, 4 or 8)"],
                  [mono("distractor_boxes, crossed_boxes"), "boxes where the other lines end / the ones whose line "
                                                            "crosses the target"],
                  [mono("rank_kept"), "yes if \"the line goes straight across\" would also give the right box "
                                      "(see below); empty for clean"],
                  [mono("objects"), "JSON: dots, boxes and every line (role target / distractor) with its control "
                                    "points. Useful for masks of the lines"]],
                 [5.6 * cm, 11.4 * cm])]

    st += [KeepTogether([P("Step 1. Get access (skip if you already did this for the other datasets)", "h"),
                         P("1. Join the <b>atml-shapes-vqa</b> organization on Hugging Face (Mohid sends the invite "
                           "link). 2. At <b>huggingface.co/settings/tokens</b> create a token of type Read. 3. In a "
                           "terminal:"),
                         Spacer(1, 3),
                         code("pip install datasets huggingface_hub pillow\n"
                              "hf auth login          # paste your token (older versions: huggingface-cli login)", w)])]
    st += [KeepTogether([P("Step 2. Load the data", "h"),
                         code(f"""from datasets import load_dataset
ds = load_dataset("{REPO}")
test = ds["test"]              # {split_imgs['test']} images = {split_scn['test']} scenes x 6 versions
ex = test[0]
ex["image"]          # PIL image, 512 x 512
ex["target_dot"]     # e.g. "B"
ex["question"]       # the prompt (already names the dot)
ex["answer"]         # "box: 5"
ex["version"]        # clean / crossings_0 / _2 / _4 / _8 / crossings_4_colored
hard = test.filter(lambda e: e["version"] == "crossings_8")    # one version only""", w)])]
    st += [KeepTogether([P("Step 3. Run your model and save its answers", "h"),
                         P(f"Same as the other datasets: a CSV of {mono('image_id, prediction')}. Always pass the "
                           "given question (it names the dot)."),
                         Spacer(1, 3),
                         code("""import csv
def ask_model(image, question):
    ...   # your VLM here (Qwen2-VL, LLaVA, InternVL, ...). Return its text answer.

with open("preds_lines_mymodel.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["image_id", "prediction"])
    for ex in test:
        w.writerow([ex["image_id"], ask_model(ex["image"], ex["question"])])""", w)])]
    st += [KeepTogether([P("Step 4. Score it", "h"),
                         code(f"""# once: download the scoring script (the same script works for all our datasets)
hf download {REPO} vqa_scoring.py --repo-type dataset --local-dir .

# score: it sees the "line_" image ids and fetches the right answers by itself
python vqa_scoring.py --predictions preds_lines_mymodel.csv --split test""", w),
                         Spacer(1, 5),
                         P("You get one row per version: <b>acc</b> (right box), <b>jumped</b> (answered the box of "
                           "a line that crosses the target, i.e. it switched lines at a crossing), <b>other</b> (box "
                           "of any other line), <b>no_line</b> (a box no line ends in), <b>parsed</b>, and accuracy "
                           "split by <b>rank_kept</b>, plus a <b>paired comparison against clean</b> with a McNemar "
                           "p-value. The parser accepts box: 5, Box 5, box five or just 5.")])]

    bullets = [
        f"Drop from {mono('clean')} to {mono('crossings_0')}: plain clutter hurts, even without a single crossing.",
        f"{mono('crossings_0')} to {mono('crossings_8')}: plot accuracy against the number of crossings. A steady "
        "fall means entanglement hurts, since nothing else changes.",
        f"High <b>jumped</b>: the model loses the line at a crossing and follows the other one.",
        f"{mono('crossings_4_colored')} vs. {mono('crossings_4')}: same lines, only the colour differs. If accuracy "
        "comes back, making the target easy to separate helps, which is what our disentanglement idea predicts.",
        f"<b>rank_kept</b>: in about 70% of the crossing images (and all of {mono('crossings_0')}) the target ends in "
        "the same position among the line ends as it starts among the dots, so guessing \"straight across\" is "
        "right there. Compare <i>acc if rank kept</i> with <i>acc if changed</i>: a model that really traces "
        "scores about the same on both; a big gap means it is guessing.",
    ]
    st += [KeepTogether([P("How to read the results", "h")] + [P("&bull; " + b, "bullet") for b in bullets])]
    st += [P("Rules of thumb", "h"),
           P("&bull; <b>Report results on the test split</b>; use train / validation for anything you tune. "
             "<b>Use the question as given</b>, so the 6 versions stay comparable. If something looks wrong, tell "
             "Mohid.", "bullet")]
    doc.build(st, onFirstPage=footer, onLaterPages=footer)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parent.parent    # repo root (this file lives in scripts/)
    ap.add_argument("--data", default=str(root / "data" / "lines_vqa"))
    ap.add_argument("--scene", type=int, default=1, help="scene shown in the example strip")
    ap.add_argument("--out", default=str(root / "docs" / "guides" / "Lines-VQA_Teammate_Guide.pdf"))
    args = ap.parse_args()
    build(Path(args.data), args.scene, Path(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
