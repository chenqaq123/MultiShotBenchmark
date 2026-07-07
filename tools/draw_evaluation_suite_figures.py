from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"

W, H = 2200, 1400

INK = (34, 43, 60)
MUTED = (91, 103, 122)
LINE = (185, 195, 210)
GRID = (229, 234, 241)
PAPER = (255, 255, 255)
BG = (248, 250, 253)

BLUE = (48, 92, 162)
TEAL = (0, 137, 122)
ORANGE = (204, 119, 44)
PURPLE = (117, 82, 160)
GREEN = (78, 139, 83)
RED = (176, 70, 68)

BLUE_FILL = (237, 245, 255)
TEAL_FILL = (235, 250, 247)
ORANGE_FILL = (255, 245, 235)
PURPLE_FILL = (245, 241, 253)
GREEN_FILL = (240, 249, 241)
GRAY_FILL = (245, 247, 250)
AGG_FILL = (250, 248, 241)


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    candidates = []
    if mono:
        candidates = [
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/Supplemental/Courier New.ttf",
        ]
    elif bold:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
        ]
    else:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


F_TITLE = font(44, bold=True)
F_SUBTITLE = font(25)
F_H = font(25, bold=True)
F_BODY = font(21)
F_SMALL = font(18)
F_TINY = font(16)
F_MONO = font(20, mono=True)
F_MONO_SMALL = font(16, mono=True)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont, max_w: int) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        if not para:
            lines.append("")
            continue
        words = para.split(" ")
        cur = ""
        for word in words:
            candidate = word if not cur else f"{cur} {word}"
            if text_size(draw, candidate, fnt)[0] <= max_w:
                cur = candidate
            else:
                if cur:
                    lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fnt: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_w: int,
    line_h: int | None = None,
    align: str = "left",
) -> int:
    x, y = xy
    lines = wrap_text(draw, text, fnt, max_w)
    if line_h is None:
        line_h = int(fnt.size * 1.25)
    yy = y
    for line in lines:
        if align == "center":
            tw, _ = text_size(draw, line, fnt)
            xx = x + (max_w - tw) / 2
        else:
            xx = x
        draw.text((xx, yy), line, font=fnt, fill=fill)
        yy += line_h
    return yy


def badge(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    label: str,
    fill: tuple[int, int, int],
    stroke: tuple[int, int, int],
    text_fill: tuple[int, int, int] = INK,
) -> int:
    pad_x = 12
    tw, th = text_size(draw, label, F_TINY)
    w = tw + 2 * pad_x
    h = 28
    draw.rounded_rectangle((x, y, x + w, y + h), radius=11, fill=fill, outline=stroke, width=1)
    draw.text((x + pad_x, y + 6), label, font=F_TINY, fill=text_fill)
    return x + w + 8


def node(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    body: str = "",
    fill: tuple[int, int, int] = PAPER,
    stroke: tuple[int, int, int] = LINE,
    title_fill: tuple[int, int, int] = INK,
    tags: list[tuple[str, tuple[int, int, int], tuple[int, int, int]]] | None = None,
    radius: int = 14,
) -> None:
    draw.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=fill, outline=stroke, width=2)
    yy = draw_wrapped(draw, (x + 22, y + 18), title, F_H, title_fill, w - 44, line_h=30) + 8
    if body:
        yy = draw_wrapped(draw, (x + 22, yy), body, F_BODY, MUTED, w - 44, line_h=28)
    if tags:
        bx = x + 22
        by = y + h - 42
        for label, tag_fill, tag_stroke in tags:
            bx = badge(draw, bx, by, label, tag_fill, tag_stroke)


def label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, color: tuple[int, int, int] = MUTED) -> None:
    draw.text((x, y), text, font=F_SMALL, fill=color)


def panel_title(draw: ImageDraw.ImageDraw, title: str, subtitle: str = "") -> None:
    draw.text((70, 46), title, font=F_TITLE, fill=INK)
    if subtitle:
        draw.text((72, 100), subtitle, font=F_SUBTITLE, fill=MUTED)


def arrow(
    draw: ImageDraw.ImageDraw,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int] = LINE,
    width: int = 4,
) -> None:
    draw.line((p1, p2), fill=color, width=width)
    angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    head = 14
    left = (
        p2[0] - head * math.cos(angle - math.pi / 7),
        p2[1] - head * math.sin(angle - math.pi / 7),
    )
    right = (
        p2[0] - head * math.cos(angle + math.pi / 7),
        p2[1] - head * math.sin(angle + math.pi / 7),
    )
    draw.polygon([p2, left, right], fill=color)


def poly_arrow(
    draw: ImageDraw.ImageDraw,
    pts: list[tuple[int, int]],
    color: tuple[int, int, int] = LINE,
    width: int = 4,
) -> None:
    for a, b in zip(pts, pts[1:]):
        draw.line((a, b), fill=color, width=width)
    arrow(draw, pts[-2], pts[-1], color=color, width=width)


def metric_box(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    items: list[str],
    accent: tuple[int, int, int],
) -> None:
    draw.rounded_rectangle((x, y, x + w, y + h), radius=14, fill=PAPER, outline=accent, width=3)
    draw.text((x + 22, y + 18), title, font=F_H, fill=accent)
    yy = y + 62
    for item in items:
        draw.ellipse((x + 24, yy + 8, x + 34, yy + 18), fill=accent)
        draw.text((x + 46, yy), item, font=F_BODY, fill=INK)
        yy += 32


def make_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    for x in range(70, W - 70, 80):
        draw.line((x, 150, x, H - 60), fill=(244, 247, 251), width=1)
    for y in range(160, H - 60, 80):
        draw.line((50, y, W - 50, y), fill=(244, 247, 251), width=1)
    return img, draw


MODEL_TAGS = {
    "llm": ("LLM / rules", PURPLE_FILL, PURPLE),
    "mllm": ("MLLM JSON", ORANGE_FILL, ORANGE),
    "det": ("OV detector", TEAL_FILL, TEAL),
    "clip": ("CLIP gate", BLUE_FILL, BLUE),
    "emb": ("DINOv2 / Face", PURPLE_FILL, PURPLE),
    "geo": ("Geometry", GREEN_FILL, GREEN),
    "agg": ("fixed formula", AGG_FILL, ORANGE),
}


def variant_1() -> Image.Image:
    img, d = make_canvas()
    panel_title(
        d,
        "Scene Continuity Evaluation Suite",
        "Algorithmic view: model calls are local; final metrics are deterministic aggregations.",
    )

    node(d, 70, 185, 300, 120, "Inputs", "generated multi-shot video\nshot-level captions", GRAY_FILL)
    node(d, 430, 185, 340, 120, "Stage 1", "shot alignment\nkeyframe sampling", PAPER, BLUE)
    arrow(d, (370, 245), (430, 245), BLUE)

    d.rounded_rectangle((70, 350, 1440, 710), radius=18, fill=(255, 255, 255), outline=(215, 226, 238), width=2)
    d.text((95, 372), "Track A: Prompt-grounded Continuity", font=F_H, fill=BLUE)
    node(d, 100, 430, 270, 180, "Caption parser", "extract active entities, actions, states, relations", PURPLE_FILL, PURPLE, tags=[MODEL_TAGS["llm"]])
    node(d, 420, 430, 280, 180, "Grounding", "locate scheduled entities and canonical crops", TEAL_FILL, TEAL, tags=[MODEL_TAGS["det"], MODEL_TAGS["clip"]])
    node(d, 750, 430, 300, 180, "Fidelity & identity", "crop embeddings plus structured semantic judge", ORANGE_FILL, ORANGE, tags=[MODEL_TAGS["emb"], MODEL_TAGS["mllm"]])
    node(d, 1100, 430, 290, 180, "PG metrics", "coverage over scheduled slots; consistency over verified opportunities", BLUE_FILL, BLUE)
    for x in [370, 700, 1050]:
        arrow(d, (x, 520), (x + 50, 520), BLUE)

    d.rounded_rectangle((70, 760, 1440, 1120), radius=18, fill=(255, 255, 255), outline=(215, 226, 238), width=2)
    d.text((95, 782), "Track B: Intrinsic Self-Consistency", font=F_H, fill=TEAL)
    node(d, 100, 840, 270, 180, "View grouping", "compare only shots with compatible visual evidence", GREEN_FILL, GREEN, tags=[MODEL_TAGS["emb"], MODEL_TAGS["geo"]])
    node(d, 420, 840, 280, 180, "Evidence mining", "global state, salient objects, coarse spatial layout", TEAL_FILL, TEAL, tags=[MODEL_TAGS["det"]])
    node(d, 750, 840, 300, 180, "Group-wise compare", "stats, embeddings, relation graphs, checkability judge", ORANGE_FILL, ORANGE, tags=[MODEL_TAGS["mllm"]])
    node(d, 1100, 840, 290, 180, "IS metrics", "richness / coverage and self-consistency", TEAL_FILL, TEAL)
    for x in [370, 700, 1050]:
        arrow(d, (x, 930), (x + 50, 930), TEAL)

    node(d, 1570, 450, 560, 430, "Deterministic aggregation", "SCS = sum_c w_c * score_c * opp_c / sum_c w_c * opp_c\n\nMLLM outputs are normalized evidence, not the final judge.", AGG_FILL, ORANGE, tags=[MODEL_TAGS["agg"]])
    arrow(d, (1390, 520), (1570, 580), BLUE)
    arrow(d, (1390, 930), (1570, 750), TEAL)

    metric_box(
        d,
        1570,
        930,
        560,
        280,
        "Outputs",
        [
            "PG-Coverage, PG-Consistency",
            "IS-Coverage / Scene Richness",
            "IS-Consistency, SCS",
            "Typed findings: Missing / Drift / Spatial / Lighting",
        ],
        ORANGE,
    )
    arrow(d, (1850, 880), (1850, 930), ORANGE)
    return img


def variant_2() -> Image.Image:
    img, d = make_canvas()
    panel_title(
        d,
        "Evaluation as Evidence Construction and Scoring",
        "The figure separates evidence extraction, model-assisted judgments, and metric computation.",
    )

    headers = [
        ("Evidence source", 300),
        ("Model / algorithm call", 650),
        ("Gate or comparison", 1080),
        ("Metric contribution", 1540),
    ]
    for title, x in headers:
        d.text((x, 185), title, font=F_H, fill=INK)
    d.line((250, 230, 2020, 230), fill=LINE, width=3)

    rows = [
        ("Prompt text", "Caption parser", "entity schedule", "PG-Coverage", PURPLE, PURPLE_FILL, [MODEL_TAGS["llm"]]),
        ("Entity crops", "Grounding + CLIP", "present / weak / absent", "PG-Consistency", BLUE, BLUE_FILL, [MODEL_TAGS["det"], MODEL_TAGS["clip"]]),
        ("Faces / objects", "DINOv2 + MLLM judge", "identity + fidelity gate", "Subject / Object scores", ORANGE, ORANGE_FILL, [MODEL_TAGS["emb"], MODEL_TAGS["mllm"]]),
        ("Background frames", "Bg embedding + features", "view groups", "valid opportunity set", GREEN, GREEN_FILL, [MODEL_TAGS["emb"], MODEL_TAGS["geo"]]),
        ("Generated scene", "Detect + layout graph", "global / object / layout", "IS-Coverage + IS-Consistency", TEAL, TEAL_FILL, [MODEL_TAGS["det"], MODEL_TAGS["mllm"]]),
    ]
    y = 280
    for source, model, gate, metric, accent, fill, tags in rows:
        d.rounded_rectangle((210, y, 2020, y + 145), radius=16, fill=PAPER, outline=(220, 228, 238), width=2)
        node(d, 265, y + 25, 285, 92, source, "", fill, accent, radius=10)
        node(d, 610, y + 25, 370, 92, model, "", fill, accent, tags=tags, radius=10)
        node(d, 1060, y + 25, 340, 92, gate, "", GRAY_FILL, accent, radius=10)
        node(d, 1510, y + 25, 390, 92, metric, "", fill, accent, radius=10)
        arrow(d, (550, y + 71), (610, y + 71), accent)
        arrow(d, (980, y + 71), (1060, y + 71), accent)
        arrow(d, (1400, y + 71), (1510, y + 71), accent)
        y += 170

    d.rounded_rectangle((210, 1150, 2020, 1300), radius=18, fill=AGG_FILL, outline=ORANGE, width=3)
    d.text((250, 1184), "Final deterministic aggregation", font=F_H, fill=ORANGE)
    d.text((250, 1230), "SCS = weighted mean over verified opportunities; coverage is reported separately to avoid empty-scene gaming.", font=F_BODY, fill=INK)
    d.text((1345, 1184), "Diagnostic outputs", font=F_H, fill=INK)
    d.text((1345, 1230), "typed findings + affected shots + evidence scores + confidence", font=F_BODY, fill=MUTED)
    return img


def variant_3() -> Image.Image:
    img, d = make_canvas()
    panel_title(
        d,
        "Two-Track Continuity Auditor",
        "Both tracks create verified comparison opportunities before any consistency score is computed.",
    )

    node(d, 80, 190, 430, 155, "Input episode", "video frames + shot captions", GRAY_FILL)
    node(d, 610, 190, 430, 155, "Shared pre-processing", "shot alignment, keyframes, masks", BLUE_FILL, BLUE)
    node(d, 1160, 190, 430, 155, "Opportunity set", "only compare visually checkable evidence", GREEN_FILL, GREEN)
    node(d, 1690, 190, 430, 155, "Score table", "coverage, consistency, SCS, findings", AGG_FILL, ORANGE)
    arrow(d, (510, 268), (610, 268), BLUE)
    arrow(d, (1040, 268), (1160, 268), GREEN)
    arrow(d, (1590, 268), (1690, 268), ORANGE)

    d.rounded_rectangle((80, 430, 1040, 1050), radius=20, fill=PAPER, outline=BLUE, width=3)
    d.text((120, 460), "Prompt-grounded branch", font=F_H, fill=BLUE)
    node(d, 130, 535, 360, 150, "Text-defined targets", "entities, actions, relations", PURPLE_FILL, PURPLE, tags=[MODEL_TAGS["llm"]])
    node(d, 585, 535, 360, 150, "Visual evidence", "bbox, crop, CLIP gate", TEAL_FILL, TEAL, tags=[MODEL_TAGS["det"], MODEL_TAGS["clip"]])
    node(d, 130, 745, 360, 150, "Fidelity gate", "exclude wrong crops", ORANGE_FILL, ORANGE, tags=[MODEL_TAGS["mllm"]])
    node(d, 585, 745, 360, 150, "Cross-shot compare", "identity, state, relation", BLUE_FILL, BLUE, tags=[MODEL_TAGS["emb"]])
    arrow(d, (490, 605), (585, 605), BLUE)
    arrow(d, (765, 675), (765, 745), BLUE)
    arrow(d, (585, 820), (490, 820), BLUE)
    arrow(d, (310, 675), (310, 745), BLUE)
    metric_box(d, 210, 935, 700, 96, "PG output", ["PG-Coverage, PG-Consistency"], BLUE)

    d.rounded_rectangle((1160, 430, 2120, 1050), radius=20, fill=PAPER, outline=TEAL, width=3)
    d.text((1200, 460), "Intrinsic scene branch", font=F_H, fill=TEAL)
    node(d, 1210, 535, 360, 150, "View grouping", "bg embedding + geometry", GREEN_FILL, GREEN, tags=[MODEL_TAGS["emb"], MODEL_TAGS["geo"]])
    node(d, 1665, 535, 360, 150, "Scene evidence", "global, objects, layout", TEAL_FILL, TEAL, tags=[MODEL_TAGS["det"]])
    node(d, 1210, 745, 360, 150, "Checkability gate", "view / occlusion aware", ORANGE_FILL, ORANGE, tags=[MODEL_TAGS["mllm"]])
    node(d, 1665, 745, 360, 150, "Group-wise compare", "state, object, relation", TEAL_FILL, TEAL)
    arrow(d, (1570, 605), (1665, 605), TEAL)
    arrow(d, (1845, 675), (1845, 745), TEAL)
    arrow(d, (1665, 820), (1570, 820), TEAL)
    arrow(d, (1390, 675), (1390, 745), TEAL)
    metric_box(d, 1290, 935, 700, 96, "IS output", ["IS-Coverage / Richness, IS-Consistency"], TEAL)

    d.rounded_rectangle((420, 1125, 1780, 1290), radius=20, fill=AGG_FILL, outline=ORANGE, width=3)
    d.text((465, 1160), "Final report", font=F_H, fill=ORANGE)
    d.text((465, 1210), "SCS aggregates verified opportunities; typed findings localize Missing, Appearance Drift, State Drift, Spatial Drift, and Lighting Drift.", font=F_BODY, fill=INK)
    arrow(d, (560, 1020), (800, 1125), BLUE)
    arrow(d, (1640, 1020), (1400, 1125), TEAL)
    return img


def variant_4() -> Image.Image:
    img, d = make_canvas()
    panel_title(
        d,
        "Where Models Enter the Evaluator",
        "A compact audit-chain view: each model call produces bounded evidence; formulas produce the leaderboard metrics.",
    )

    phases = [
        ("1. Build evidence", 120, 270, BLUE),
        ("2. Gate opportunities", 705, 270, GREEN),
        ("3. Compare consistency", 1290, 270, TEAL),
        ("4. Aggregate metrics", 1810, 270, ORANGE),
    ]
    for title, x, y, color in phases:
        d.text((x, 190), title, font=F_H, fill=color)
        d.line((x, 235, x + 360, 235), fill=color, width=4)

    node(d, 90, 300, 430, 210, "Prompt branch evidence", "caption entities -> entity crops -> canonical crops", BLUE_FILL, BLUE, tags=[MODEL_TAGS["llm"], MODEL_TAGS["det"], MODEL_TAGS["clip"]])
    node(d, 90, 615, 430, 210, "Intrinsic branch evidence", "view groups -> global state -> salient objects -> layout graph", TEAL_FILL, TEAL, tags=[MODEL_TAGS["det"], MODEL_TAGS["emb"], MODEL_TAGS["geo"]])

    node(d, 675, 300, 430, 210, "Prompt gates", "presence, crop quality, entity fidelity, action validity", ORANGE_FILL, ORANGE, tags=[MODEL_TAGS["mllm"]])
    node(d, 675, 615, 430, 210, "Intrinsic gates", "view comparability, occlusion, partial view, checkability", GREEN_FILL, GREEN, tags=[MODEL_TAGS["mllm"]])

    node(d, 1260, 300, 430, 210, "Prompt comparisons", "identity, appearance, state, action, explicit relation", BLUE_FILL, BLUE, tags=[MODEL_TAGS["emb"], MODEL_TAGS["mllm"]])
    node(d, 1260, 615, 430, 210, "Intrinsic comparisons", "global state similarity, object match, layout relation agreement", TEAL_FILL, TEAL, tags=[MODEL_TAGS["emb"], MODEL_TAGS["mllm"]])

    node(d, 1815, 365, 300, 355, "Scores", "PG-Coverage\nPG-Consistency\nIS-Coverage\nIS-Consistency\nSCS", AGG_FILL, ORANGE, tags=[MODEL_TAGS["agg"]])

    for y in [405, 720]:
        arrow(d, (520, y), (675, y), LINE)
        arrow(d, (1105, y), (1260, y), LINE)
        arrow(d, (1690, y), (1815, 545), LINE)

    d.rounded_rectangle((150, 955, 2050, 1265), radius=22, fill=PAPER, outline=(215, 226, 238), width=2)
    d.text((200, 990), "Model-use legend", font=F_H, fill=INK)
    legend = [
        ("LLM / rules", "parse caption into auditable prompt entity list", PURPLE_FILL, PURPLE),
        ("Detector + CLIP", "localize scheduled entities and salient scene objects", TEAL_FILL, TEAL),
        ("DINOv2 / ArcFace", "compute appearance and identity similarities", PURPLE_FILL, PURPLE),
        ("Geometry / RANSAC", "form comparable view groups before intrinsic comparison", GREEN_FILL, GREEN),
        ("MLLM JSON", "semantic fidelity, checkability, same/different, typed explanation", ORANGE_FILL, ORANGE),
        ("Fixed formula", "aggregate scores over verified opportunities", AGG_FILL, ORANGE),
    ]
    x, y = 200, 1050
    for name, desc, fill, stroke in legend:
        badge(d, x, y, name, fill, stroke)
        d.text((x + 190, y + 4), desc, font=F_SMALL, fill=MUTED)
        y += 36

    d.text((1270, 990), "Typed findings", font=F_H, fill=INK)
    findings = ["Missing", "Appearance Drift", "State Drift", "Spatial Drift", "Lighting / Atmosphere Drift"]
    yy = 1050
    for item in findings:
        d.rounded_rectangle((1270, yy, 1645, yy + 32), radius=10, fill=GRAY_FILL, outline=LINE, width=1)
        d.text((1288, yy + 5), item, font=F_SMALL, fill=INK)
        yy += 42
    draw_wrapped(
        d,
        (1710, 1058),
        "Each finding stores element, affected shots, evidence scores, confidence, and severity.",
        F_BODY,
        MUTED,
        300,
        line_h=30,
    )
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    variants = [variant_1(), variant_2(), variant_3(), variant_4()]
    for idx, img in enumerate(variants, start=1):
        out = OUT / f"evaluation_suite_algorithm_variant_{idx}.png"
        img.save(out, quality=95)
        print(out)


if __name__ == "__main__":
    main()
