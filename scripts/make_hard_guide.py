#!/usr/bin/env python
"""
Build docs/guides/Shapes-VQA_Hard_Teammate_Guide.pdf (same layout as the other teammate guides, red accent).

Needs the built dataset (data/shapes_vqa_hard/) for the example images and reportlab:
    python scripts/make_hard_guide.py [--data data/shapes_vqa_hard] [--scene 1]
                                      [--out docs/guides/Shapes-VQA_Hard_Teammate_Guide.pdf]
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

ACCENT = colors.HexColor("#b91c1c")       # the colours of the original v1.1 guide
ACCENT_BG = colors.HexColor("#fef2f2")
GREY_BG = colors.HexColor("#f3f4f6")
LINE = colors.HexColor("#d1d5db")
TEXT = colors.HexColor("#1f2937")
MUTED = colors.HexColor("#6b7280")
REPO = "atml-shapes-vqa/shapes-vqa-hard"
STRIP_ROWS = [["clean", "wrong_color", "lookalikes", "text_traps", "everything"],
              ["squiggles_2", "squiggles_4", "squiggles_8", "squiggles_16", "other_shapes"]]
N_VERSIONS = sum(len(r) for r in STRIP_ROWS)

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
    """One scene in all 10 versions (2 rows x 5), with labels and the answer, as a reportlab Image."""
    rows = {r["version"]: r for r in csv.DictReader(open(root / "metadata.csv", encoding="utf-8"))
            if int(r["scene_id"]) == scene}
    tile, pad, head, foot = 300, 10, 28, 32
    W = pad + 5 * (tile + pad)
    H = pad + 2 * (head + tile + pad) + foot
    sheet = PILImage.new("RGB", (W, H), (236, 236, 236))
    d = ImageDraw.Draw(sheet)
    f = ImageFont.load_default(size=18)
    for ri, versions in enumerate(STRIP_ROWS):
        for ci, v in enumerate(versions):
            x, y = pad + ci * (tile + pad), pad + ri * (head + tile + pad)
            with PILImage.open(root / rows[v]["file_name"]) as im:
                sheet.paste(im.convert("RGB").resize((tile, tile), PILImage.LANCZOS), (x, y + head))
            d.text((x, y + 5), f"scene {scene}  |  {v}", fill=(0, 0, 0), font=f)
    r = rows["clean"]
    d.text((pad, H - foot + 6), f"count the {r['target_color'].upper()} shapes  ->  answer (all versions): "
           f"{r['answer']}   [{r['split']}]", fill=(0, 0, 0), font=f)
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
    canvas.drawString(2 * cm, 1.2 * cm, "Shapes-VQA Hard - teammate guide")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"page {doc.page}")
    canvas.restoreState()


def build(root, scene, out):
    rows = list(csv.DictReader(open(root / "metadata.csv", encoding="utf-8")))
    versions_found = list(dict.fromkeys(r["version"] for r in rows))
    if sorted(versions_found) != sorted(v for row in STRIP_ROWS for v in row):
        raise SystemExit(f"{root} has versions {versions_found}; this guide describes {STRIP_ROWS}")
    split_imgs = {s: sum(r["split"] == s for r in rows) for s in ("train", "validation", "test")}
    split_scn = {s: n // N_VERSIONS for s, n in split_imgs.items()}

    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=2 * cm, title="Shapes-VQA Hard: How to Use It", author="Mohid Ali")
    w = A4[0] - 4 * cm
    st = []
    st += [P("Shapes-VQA Hard: How to Use It", "title"), Spacer(1, 2),
           P(f"ATML project&nbsp;&nbsp;|&nbsp;&nbsp;huggingface.co/datasets/{REPO}&nbsp;&nbsp;|&nbsp;&nbsp;questions: ask Mohid", "sub")]

    st += [P("What it is", "h"),
           P("The harder version of our dataset: count only the named shapes (circle, triangle, square, star) "
             f"<b>of one colour</b>, named in the question. <b>{len(rows) // N_VERSIONS} scenes x {N_VERSIONS} versions "
             f"= {len(rows):,} images</b>. The traps target <b>mixing up colours and shapes</b>, <b>being fooled by "
             "text in the image</b>, <b>squiggles of growing complexity</b>, and <b>clutter with a name vs. clutter "
             "without one</b>."),
           Spacer(1, 6),
           table([["Version", "Diff.", "What is added to the target-colour shapes", "What it tests"],
                  ["clean", "0", "nothing", "baseline"],
                  ["wrong_color", "1", "perfect named shapes in <b>other</b> colours", "colour-shape binding"],
                  ["lookalikes", "2", "broken or wavy shapes in the <b>target</b> colour", "shape similarity"],
                  ["text_traps", "3", "misleading text like \"star: 4\", \"red circle\", \"5\" (count claims are "
                                      "always wrong)", "being fooled by text"],
                  ["everything", "4", "all traps from versions 1, 2 and 3 together", "worst case"],
                  ["squiggles_2, _4, _8, _16", "5-8", "scribbles with 2, 4, 8 or 16 curves (a curve = one bend of "
                   "30-120 degrees), in the <b>same spots, sizes and colours</b> in all four", "squiggle complexity"],
                  ["other_shapes", "9", "<b>nameable</b> shapes that are none of the four types (heart, moon, cross, "
                   "lightning bolt), in <b>the squiggle spots, sizes and colours</b>", "clutter with vs. without a "
                   "name"]],
                 [2.9 * cm, 1.2 * cm, 8.8 * cm, 4.1 * cm]),
           Spacer(1, 8),
           callout(f"<b>Key idea:</b> the target shapes are pixel-identical in all {N_VERSIONS} versions of a scene, "
                   f"so <b>all {N_VERSIONS} versions have the same correct answer</b>. Each version adds one kind of "
                   "trap, so each version's accuracy drop tells you <b>which trap</b> hurts the model most. The 4 "
                   "squiggle versions change <b>only</b> the number of curves, so they give accuracy vs. complexity. "
                   f"{mono('other_shapes')} draws nameable shapes in exactly the squiggle spots, so comparing it with "
                   "the squiggle versions isolates <b>whether the clutter has a name</b>.", w),
           Spacer(1, 8),
           example_strip(root, scene, w),
           Spacer(1, 6),
           KeepTogether([P("The question is the same for every image except the colour (red, blue, green or black):"),
                         Spacer(1, 3),
                         code("... Only count complete, regular shapes that are red and are one of these four types:\n"
                              "circle, triangle, square, star. Do not count shapes of any other colour, broken or wavy\n"
                              "shapes, or any words or numbers written in the image. Answer in exactly this format: ...\n\n"
                              "answer:  circle: 1, triangle: 0, square: 2, star: 1      (all 4 classes, zeros included)", w)]),
           Spacer(1, 5),
           P(f"Versions 1-3, each squiggle version and {mono('other_shapes')} add 2, 3 or 4 traps; <i>everything</i> "
             "has exactly the union of versions 1-3 (6, 9 or 12). None of the other shapes is or contains a circle, "
             "triangle, square or star, so they are never counted. Splits are by scene: <b>train</b> "
             f"{split_imgs['train']:,} images ({split_scn['train']} scenes), <b>validation</b> {split_imgs['validation']} "
             f"({split_scn['validation']}), <b>test</b> {split_imgs['test']} ({split_scn['test']}). "
             f"{mono('other_shapes')} was added in v1.2; the other 9 versions are byte-identical to before, so "
             "results you already have stay valid.")]

    st += [P("Useful columns", "h"),
           table([["Column", "Meaning"],
                  [mono("image_id"), "e.g. hard_0042_text_traps. The key for your predictions"],
                  [mono("target_color"), "the colour to count: red / blue / green / black"],
                  [mono("scene_id"), f"same for all {N_VERSIONS} versions of a scene. Use it to compare versions"],
                  [mono("version, difficulty"), "clean / wrong_color / lookalikes / text_traps / everything = 0-4, "
                                                "squiggles_2 / _4 / _8 / _16 = 5-8, other_shapes = 9"],
                  [mono("count_&lt;shape&gt;, count_total"), "true counts (target colour only)"],
                  [mono("objects"), "JSON list of every object; traps have role = distractor. Lookalikes have "
                                    "lookalike_of and subtype gap / wavy, text traps the written text, squiggles "
                                    "curves and turning_deg (total bending), other shapes shape = heart / moon / "
                                    "cross / bolt"]],
                 [5.0 * cm, 12.0 * cm])]

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
test = ds["test"]            # {split_imgs['test']} images = {split_scn['test']} scenes x {N_VERSIONS} versions
ex = test[0]
ex["image"]          # PIL image, 512 x 512
ex["target_color"]   # e.g. "red"
ex["question"]       # the prompt (already names the colour)
ex["answer"]         # "circle: 1, triangle: 0, square: 2, star: 1"
ex["version"]        # clean / wrong_color / ... / squiggles_16 / other_shapes
text_only = test.filter(lambda e: e["version"] == "text_traps")   # one version only""", w)])]
    st += [KeepTogether([P("Step 3. Run your model and save its answers", "h"),
                         P(f"Same as the other datasets: a CSV of {mono('image_id, prediction')}. Always pass the "
                           "given question (it names the colour)."),
                         Spacer(1, 3),
                         code("""import csv
def ask_model(image, question):
    ...   # your VLM here (Qwen2-VL, LLaVA, InternVL, ...). Return its text answer.

with open("preds_hard_mymodel.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["image_id", "prediction"])
    for ex in test:
        w.writerow([ex["image_id"], ask_model(ex["image"], ex["question"])])""", w)])]
    st += [KeepTogether([P("Step 4. Score it", "h"),
                         code(f"""# once: download the scoring script (the same script works for all our datasets)
hf download {REPO} vqa_scoring.py --repo-type dataset --local-dir .

# score: it sees the "hard_" image ids and fetches the right answers by itself
python vqa_scoring.py --predictions preds_hard_mymodel.csv --split test""", w),
                         Spacer(1, 5),
                         P("You get one row per version: <b>exact</b> (all 4 counts right), per-class accuracy, "
                           "<b>abs_err</b> (summed count error), <b>overcnt</b> (share of images where the model "
                           "counted more target shapes than there are) and <b>parsed</b>, plus a <b>paired comparison "
                           "against clean</b> with a McNemar p-value. Add <font name=\"Courier\">--out scores.csv</font> "
                           "to get the per-image scores.")])]

    bullets = [
        f"Big drop on {mono('wrong_color')}: the model mixes up colours and shapes (binding problem).",
        f"Big drop on {mono('lookalikes')}: it can't tell near-miss shapes from real ones.",
        f"Big drop on {mono('text_traps')}: it reads the text and copies its numbers (the written words are in "
        f"{mono('objects')}).",
        f"{mono('everything')}: do the traps add up, or is one trap enough to break it?",
        f"{mono('squiggles_2')} to {mono('squiggles_16')}: plot accuracy against the number of curves. A steady fall "
        "means more complex clutter hurts more, even though nothing else changes.",
        f"{mono('other_shapes')} vs. the squiggle versions: same spots, sizes, colours and number of traps; only "
        "whether the clutter has a name differs. A bigger drop on other_shapes means clutter the model can name "
        "distracts it more. Compare on the same scenes (paired by scene_id).",
    ]
    st += [KeepTogether([P("How to read the results", "h")] + [P("&bull; " + b, "bullet") for b in bullets])]
    st += [P("Rules of thumb", "h"),
           P("&bull; <b>Report results on the test split</b>; use train / validation for anything you tune. "
             f"<b>Use the question as given</b>, so the {N_VERSIONS} versions stay comparable. If something looks "
             "wrong, tell Mohid.", "bullet")]
    doc.build(st, onFirstPage=footer, onLaterPages=footer)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parent.parent    # repo root (this file lives in scripts/)
    ap.add_argument("--data", default=str(root / "data" / "shapes_vqa_hard"))
    ap.add_argument("--scene", type=int, default=1, help="scene shown in the example strip")
    ap.add_argument("--out", default=str(root / "docs" / "guides" / "Shapes-VQA_Hard_Teammate_Guide.pdf"))
    args = ap.parse_args()
    build(Path(args.data), args.scene, Path(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
