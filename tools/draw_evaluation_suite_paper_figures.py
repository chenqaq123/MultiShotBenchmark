from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"

W, H = 2400, 1500
PAPER = (255, 255, 255)
INK = (23, 27, 35)
MID = (82, 92, 108)
FAINT = (218, 224, 233)
LIGHT = (246, 248, 251)

BLUE = (36, 88, 157)
TEAL = (0, 126, 116)
ORANGE = (190, 104, 34)
PURPLE = (104, 78, 150)
GREEN = (78, 126, 72)


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
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


F_TITLE = font(42, bold=True)
F_SUB = font(24)
F_PANEL = font(28, bold=True)
F_H = font(24, bold=True)
F_BODY = font(20)
F_SMALL = font(17)
F_TINY = font(15)
F_MONO = font(19, mono=True)
F_MONO_SMALL = font(16, mono=True)


def size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont, max_w: int) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        words = para.split(" ")
        cur = ""
        for word in words:
            cand = word if not cur else f"{cur} {word}"
            if size(draw, cand, fnt)[0] <= max_w:
                cur = cand
            else:
                if cur:
                    lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
    return lines


def text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    content: str,
    fnt: ImageFont.ImageFont,
    color: tuple[int, int, int] = INK,
    max_w: int | None = None,
    line_h: int | None = None,
) -> int:
    x, y = xy
    if max_w is None:
        draw.text((x, y), content, font=fnt, fill=color)
        return y + size(draw, content, fnt)[1]
    if line_h is None:
        line_h = int(fnt.size * 1.35)
    yy = y
    for line in wrap(draw, content, fnt, max_w):
        draw.text((x, yy), line, font=fnt, fill=color)
        yy += line_h
    return yy


def canvas(title: str, subtitle: str = "") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), PAPER)
    draw = ImageDraw.Draw(img)
    draw.text((80, 54), title, font=F_TITLE, fill=INK)
    if subtitle:
        draw.text((82, 108), subtitle, font=F_SUB, fill=MID)
    draw.line((80, 158, W - 80, 158), fill=FAINT, width=2)
    return img, draw


def box(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    body: str = "",
    edge: tuple[int, int, int] = INK,
    fill: tuple[int, int, int] = PAPER,
    lw: int = 2,
    title_color: tuple[int, int, int] | None = None,
) -> None:
    draw.rectangle((x, y, x + w, y + h), fill=fill, outline=edge, width=lw)
    yy = text(draw, (x + 18, y + 15), title, F_H, title_color or INK, max_w=w - 36, line_h=28) + 7
    if body:
        text(draw, (x + 18, yy), body, F_BODY, MID, max_w=w - 36, line_h=27)


def small_tag(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    label: str,
    color: tuple[int, int, int],
) -> int:
    tw, th = size(draw, label, F_TINY)
    pad = 8
    draw.rounded_rectangle((x, y, x + tw + 2 * pad, y + 24), radius=4, fill=PAPER, outline=color, width=1)
    draw.text((x + pad, y + 5), label, font=F_TINY, fill=color)
    return x + tw + 2 * pad + 8


def model_tags(draw: ImageDraw.ImageDraw, x: int, y: int, labels: list[tuple[str, tuple[int, int, int]]]) -> None:
    xx = x
    for label, color in labels:
        xx = small_tag(draw, xx, y, label, color)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int] = INK,
    width: int = 2,
) -> None:
    draw.line((start, end), fill=color, width=width)
    ang = math.atan2(end[1] - start[1], end[0] - start[0])
    head = 12
    p1 = (end[0] - head * math.cos(ang - math.pi / 7), end[1] - head * math.sin(ang - math.pi / 7))
    p2 = (end[0] - head * math.cos(ang + math.pi / 7), end[1] - head * math.sin(ang + math.pi / 7))
    draw.polygon([end, p1, p2], fill=color)


def polyline(draw: ImageDraw.ImageDraw, pts: list[tuple[int, int]], color: tuple[int, int, int], width: int = 2) -> None:
    for a, b in zip(pts, pts[1:]):
        draw.line((a, b), fill=color, width=width)
    arrow(draw, pts[-2], pts[-1], color=color, width=width)


def bracket(draw: ImageDraw.ImageDraw, x: int, y1: int, y2: int, label: str, color: tuple[int, int, int]) -> None:
    draw.line((x, y1, x, y2), fill=color, width=3)
    draw.line((x, y1, x + 24, y1), fill=color, width=3)
    draw.line((x, y2, x + 24, y2), fill=color, width=3)
    draw.text((x + 34, y1 - 17), label, font=F_PANEL, fill=color)


def equation_block(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, title: str, lines: list[str]) -> None:
    draw.rectangle((x, y, x + w, y + h), fill=(254, 253, 249), outline=ORANGE, width=2)
    draw.text((x + 20, y + 18), title, font=F_H, fill=ORANGE)
    yy = y + 64
    for line in lines:
        draw.text((x + 24, yy), line, font=F_MONO, fill=INK)
        yy += 36


def legend(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    entries = [
        ("LLM/rules", PURPLE, "caption parsing"),
        ("OVDet+CLIP", TEAL, "grounding / filtering"),
        ("DINOv2/Face", BLUE, "embedding similarity"),
        ("Geom.", GREEN, "view grouping"),
        ("MLLM", ORANGE, "structured semantic judge"),
        ("Formula", INK, "deterministic aggregation"),
    ]
    draw.text((x, y), "Model-use legend", font=F_H, fill=INK)
    yy = y + 40
    for name, col, desc in entries:
        small_tag(draw, x, yy, name, col)
        draw.text((x + 132, yy + 3), desc, font=F_SMALL, fill=MID)
        yy += 31


def variant_1() -> Image.Image:
    img, d = canvas(
        "Evaluation suite for multi-shot scene continuity",
        "Paper-style method overview: local model calls, verified opportunities, deterministic metrics.",
    )

    box(d, 90, 220, 300, 108, "Input", "video V\nshot captions C", edge=INK)
    box(d, 470, 220, 330, 108, "Shot alignment", "boundaries, keyframes K", edge=BLUE)
    arrow(d, (390, 274), (470, 274), INK)

    bracket(d, 96, 430, 735, "A. Prompt-grounded continuity", BLUE)
    x0, y0, bw, bh, gap = 170, 490, 292, 135, 54
    nodes = [
        ("Entity schedule", "E_p = Parse(C)\nactive entities / relations", PURPLE, [("LLM/rules", PURPLE)]),
        ("Grounded evidence", "bbox, crop, visibility status", TEAL, [("OVDet+CLIP", TEAL)]),
        ("Fidelity gate", "entity/action validity\nexclude wrong crops", ORANGE, [("MLLM", ORANGE)]),
        ("Cross-shot compare", "identity, appearance,\nstate, relation", BLUE, [("DINOv2/Face", BLUE), ("MLLM", ORANGE)]),
    ]
    for i, (t, b, c, tags) in enumerate(nodes):
        x = x0 + i * (bw + gap)
        box(d, x, y0, bw, bh, t, b, edge=c)
        model_tags(d, x + 18, y0 + bh - 34, tags)
        if i:
            arrow(d, (x - gap, y0 + bh // 2), (x, y0 + bh // 2), BLUE)

    box(d, 1610, 490, 320, 135, "PG scores", "PG-Coverage\nPG-Consistency", edge=BLUE)
    arrow(d, (x0 + 4 * bw + 3 * gap, y0 + bh // 2), (1610, y0 + bh // 2), BLUE)

    bracket(d, 96, 810, 1115, "B. Intrinsic self-consistency", TEAL)
    y1 = 870
    nodes2 = [
        ("View groups", "comparable-shot groups G", GREEN, [("DINOv2", BLUE), ("Geom.", GREEN)]),
        ("Scene evidence", "global state, objects,\nlayout graph", TEAL, [("OVDet", TEAL)]),
        ("Checkability", "view / occlusion aware", ORANGE, [("MLLM", ORANGE)]),
        ("Group-wise compare", "state, object match,\nlayout relation", TEAL, [("DINOv2", BLUE), ("MLLM", ORANGE)]),
    ]
    for i, (t, b, c, tags) in enumerate(nodes2):
        x = x0 + i * (bw + gap)
        box(d, x, y1, bw, bh, t, b, edge=c)
        model_tags(d, x + 18, y1 + bh - 34, tags)
        if i:
            arrow(d, (x - gap, y1 + bh // 2), (x, y1 + bh // 2), TEAL)

    box(d, 1610, 870, 320, 135, "IS scores", "IS-Coverage / Richness\nIS-Consistency", edge=TEAL)
    arrow(d, (x0 + 4 * bw + 3 * gap, y1 + bh // 2), (1610, y1 + bh // 2), TEAL)

    equation_block(
        d,
        1985,
        575,
        330,
        370,
        "Final scorer",
        [
            "SCS = sum_c w_c *",
            "      score_c * opp_c",
            "      / sum_c w_c * opp_c",
            "",
            "Report: coverage +",
            "consistency + findings",
        ],
    )
    arrow(d, (1930, 556), (1985, 650), BLUE)
    arrow(d, (1930, 936), (1985, 870), TEAL)

    legend(d, 145, 1240)
    text(d, (780, 1244), "Typed findings: Missing, Appearance Drift, State Drift, Spatial Drift, Lighting / Atmosphere Drift.", F_BODY, MID)
    return img


def variant_2() -> Image.Image:
    img, d = canvas(
        "Evidence-to-metric computation graph",
        "Each row is a bounded evidence path; final leaderboard numbers are computed by fixed formulas.",
    )

    left, top = 130, 250
    col_w = [300, 410, 410, 420]
    headers = ["Evidence", "Model / algorithm", "Gate or comparison", "Metric"]
    xs = [left]
    for w in col_w[:-1]:
        xs.append(xs[-1] + w + 38)
    for x, w, htxt in zip(xs, col_w, headers):
        d.text((x + 6, top - 56), htxt, font=F_PANEL, fill=INK)
        d.line((x, top - 18, x + w, top - 18), fill=INK, width=2)

    rows = [
        ("caption C", "Parse active entities", "entity schedule", "PG-Coverage", PURPLE, [("LLM/rules", PURPLE)]),
        ("entity crops", "Grounding + CLIP", "present / weak / absent", "PG-Consistency", BLUE, [("OVDet", TEAL), ("CLIP", BLUE)]),
        ("faces / objects", "Embedding + semantic judge", "identity + fidelity", "Subject / Object", ORANGE, [("DINOv2/Face", BLUE), ("MLLM", ORANGE)]),
        ("background frames", "Bg embedding + geometry", "view groups", "opportunity set", GREEN, [("DINOv2", BLUE), ("Geom.", GREEN)]),
        ("generated scene", "Detect + layout graph", "global / object / layout", "IS-Coverage + IS-Consistency", TEAL, [("OVDet", TEAL), ("MLLM", ORANGE)]),
    ]
    y = top
    row_h = 142
    for evidence, model, gate, metric, color, tags in rows:
        d.rectangle((left - 30, y - 20, left + sum(col_w) + 3 * 38 + 30, y + row_h - 20), outline=FAINT, width=2)
        box(d, xs[0], y, col_w[0], 82, evidence, edge=color, lw=2)
        box(d, xs[1], y, col_w[1], 82, model, edge=color, lw=2)
        model_tags(d, xs[1] + 18, y + 52, tags)
        box(d, xs[2], y, col_w[2], 82, gate, edge=color, lw=2)
        box(d, xs[3], y, col_w[3], 82, metric, edge=color, lw=2)
        arrow(d, (xs[0] + col_w[0], y + 41), (xs[1], y + 41), color)
        arrow(d, (xs[1] + col_w[1], y + 41), (xs[2], y + 41), color)
        arrow(d, (xs[2] + col_w[2], y + 41), (xs[3], y + 41), color)
        y += row_h + 32

    equation_block(
        d,
        130,
        1190,
        1005,
        180,
        "Deterministic aggregation",
        [
            "score_c is computed only over verified opportunities opp_c",
            "SCS = weighted_mean_c(score_c, weights = w_c * opp_c)",
        ],
    )
    box(
        d,
        1230,
        1190,
        900,
        180,
        "Diagnostic report",
        "metric vector: [PG-Cov, PG-Con, IS-Cov, IS-Con, SCS]\nfindings: element, error type, affected shots, evidence, confidence",
        edge=INK,
    )
    return img


def variant_3() -> Image.Image:
    img, d = canvas(
        "Opportunity-normalized continuity scoring",
        "The suite first defines checkable opportunities, then computes prompt-grounded and intrinsic consistency.",
    )

    box(d, 110, 250, 420, 130, "Episode inputs", "V: generated video\nC: shot-level captions", edge=INK)
    box(d, 660, 250, 420, 130, "Preprocess", "shot alignment, keyframes,\nforeground masks", edge=BLUE)
    box(d, 1210, 250, 420, 130, "Evidence store", "crops, embeddings,\nview groups, layout graphs", edge=TEAL)
    box(d, 1760, 250, 420, 130, "Verified opportunity set", "only score observable and comparable evidence", edge=GREEN)
    for x in [530, 1080, 1630]:
        arrow(d, (x, 315), (x + 130, 315), INK)

    d.line((310, 510, 310, 1065), fill=FAINT, width=3)
    d.line((1195, 510, 1195, 1065), fill=FAINT, width=3)
    d.line((2050, 510, 2050, 1065), fill=FAINT, width=3)

    d.text((140, 520), "Prompt-grounded track", font=F_PANEL, fill=BLUE)
    d.text((1025, 520), "Comparators and gates", font=F_PANEL, fill=INK)
    d.text((1880, 520), "Metric terms", font=F_PANEL, fill=ORANGE)

    pg_items = [
        ("Scheduled entity slots", "characters, objects, actions, relations", PURPLE, [("LLM/rules", PURPLE)]),
        ("Canonical crops", "localized by prompt entity queries", TEAL, [("OVDet+CLIP", TEAL)]),
    ]
    is_items = [
        ("Comparable view groups", "same or partially shared scene view", GREEN, [("DINOv2", BLUE), ("Geom.", GREEN)]),
        ("Intrinsic scene evidence", "global state, salient objects, layout", TEAL, [("OVDet", TEAL)]),
    ]
    y = 610
    for title, body, color, tags in pg_items:
        box(d, 120, y, 520, 112, title, body, edge=color)
        model_tags(d, 330, y + 77, tags)
        y += 145
    y = 610
    for title, body, color, tags in is_items:
        box(d, 120, y + 310, 520, 112, title, body, edge=color)
        model_tags(d, 330, y + 387, tags)
        y += 145

    comp = [
        ("Presence / fidelity", "tri-state visibility, crop fidelity, action validity", ORANGE, [("MLLM", ORANGE)]),
        ("Appearance / state", "embedding similarity, face identity, relation stability", BLUE, [("DINOv2/Face", BLUE)]),
        ("Scene consistency", "global stats, object match, layout relation agreement", TEAL, [("MLLM", ORANGE)]),
    ]
    y = 640
    for title, body, color, tags in comp:
        box(d, 805, y, 600, 120, title, body, edge=color)
        model_tags(d, 1040, y + 83, tags)
        y += 180

    metric_y = 620
    metric_lines = [
        ("PG-Coverage", "# present slots / # scheduled slots", BLUE),
        ("PG-Consistency", "mean stable prompt elements", BLUE),
        ("IS-Coverage", "# verifiable scene evidence", TEAL),
        ("IS-Consistency", "mean stable intrinsic evidence", TEAL),
        ("SCS", "opportunity-weighted aggregate", ORANGE),
    ]
    for name, desc, color in metric_lines:
        box(d, 1640, metric_y, 550, 80, name, desc, edge=color)
        metric_y += 96

    for sy, ey in [(666, 700), (811, 880), (976, 1060)]:
        polyline(d, [(640, sy), (725, sy), (725, ey), (805, ey)], INK)
        polyline(d, [(1405, ey), (1515, ey), (1515, ey - 20), (1640, ey - 20)], INK)

    equation_block(
        d,
        805,
        1190,
        930,
        150,
        "Scalar score, when ranking is required",
        ["SCS = sum_c w_c * score_c * opp_c / sum_c w_c * opp_c"],
    )
    text(d, (1775, 1215), "Coverage is always reported with consistency, so sparse videos cannot obtain high scores by avoiding checkable content.", F_BODY, MID, max_w=430)
    return img


def variant_4() -> Image.Image:
    img, d = canvas(
        "Algorithmic audit chain for continuity diagnosis",
        "A compact figure emphasizing which model is used at each step and what metric is produced.",
    )

    d.text((110, 225), "(a) Structured evaluator", font=F_PANEL, fill=INK)
    box(d, 110, 270, 900, 840, "Algorithm 1: SceneContinuityEval(V, C)", "", edge=INK)
    code = [
        "1  K <- AlignShotsAndSampleKeyframes(V, C)",
        "2  E_p <- ParseCaptionEntities(C)                  [LLM/rules]",
        "3  X_p <- GroundPromptEntities(K, E_p)              [OVDet+CLIP]",
        "4  Q_p <- JudgePromptFidelity(X_p, C)               [MLLM]",
        "5  S_pg <- ComparePromptEvidence(X_p, Q_p)          [DINOv2/Face]",
        "",
        "6  G <- GroupComparableViews(K)                     [DINOv2+Geom.]",
        "7  Z <- MineIntrinsicEvidence(K, G)                 [OVDet]",
        "8  Q_i <- JudgeCheckability(Z, G)                   [MLLM]",
        "9  S_is <- CompareSceneEvidence(Z, Q_i)             [stats/graph]",
        "",
        "10 metrics <- Aggregate(S_pg, S_is, opp)            [Formula]",
        "11 findings <- LocalizeTypedErrors(S_pg, S_is)",
    ]
    yy = 340
    for line in code:
        col = MID
        if "[MLLM]" in line:
            col = ORANGE
        elif "[OVDet" in line:
            col = TEAL
        elif "[DINO" in line:
            col = BLUE
        elif "[Formula]" in line:
            col = INK
        d.text((150, yy), line, font=F_MONO_SMALL, fill=col)
        yy += 48 if line else 24

    d.text((1170, 225), "(b) Metric-producing paths", font=F_PANEL, fill=INK)
    box(d, 1160, 295, 470, 128, "Prompt path", "caption targets -> grounded crops -> fidelity gate -> cross-shot compare", edge=BLUE)
    model_tags(d, 1180, 385, [("LLM/rules", PURPLE), ("OVDet+CLIP", TEAL), ("MLLM", ORANGE), ("DINOv2", BLUE)])
    box(d, 1790, 295, 420, 128, "Prompt metrics", "PG-Coverage\nPG-Consistency", edge=BLUE)
    arrow(d, (1630, 359), (1790, 359), BLUE)

    box(d, 1160, 535, 470, 128, "Intrinsic path", "view groups -> scene evidence -> checkability -> group-wise compare", edge=TEAL)
    model_tags(d, 1180, 625, [("DINOv2+Geom.", GREEN), ("OVDet", TEAL), ("MLLM", ORANGE)])
    box(d, 1790, 535, 420, 128, "Intrinsic metrics", "IS-Coverage / Richness\nIS-Consistency", edge=TEAL)
    arrow(d, (1630, 599), (1790, 599), TEAL)

    box(
        d,
        1160,
        800,
        470,
        210,
        "Typed findings",
        "Missing\nAppearance Drift\nState Drift\nSpatial Drift\nLighting Drift",
        edge=ORANGE,
    )
    equation_block(
        d,
        1790,
        800,
        420,
        210,
        "Aggregation",
        [
            "metric vector m =",
            "[PG-Cov, PG-Con,",
            " IS-Cov, IS-Con, SCS]",
            "SCS weighted by opp_c",
        ],
    )
    polyline(d, [(2210, 359), (2260, 359), (2260, 860), (2210, 860)], BLUE)
    polyline(d, [(2210, 599), (2240, 599), (2240, 930), (2210, 930)], TEAL)
    arrow(d, (1790, 905), (1630, 905), ORANGE)

    legend(d, 1160, 1185)
    text(d, (1760, 1190), "Design principle: MLLM provides structured local judgments and explanations; the leaderboard score is produced by fixed aggregation over verified opportunities.", F_BODY, MID, max_w=500)
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    variants = [variant_1(), variant_2(), variant_3(), variant_4()]
    for idx, img in enumerate(variants, start=1):
        path = OUT / f"evaluation_suite_paper_variant_{idx}.png"
        img.save(path, quality=96)
        print(path)


if __name__ == "__main__":
    main()
