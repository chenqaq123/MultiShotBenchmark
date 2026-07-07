from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"

W, H = 2400, 1350

WHITE = (255, 255, 255)
INK = (20, 24, 31)
MUTED = (75, 86, 104)
GRAY = (92, 92, 92)

BLUE = (49, 105, 190)
TEAL = (0, 151, 142)
ORANGE = (242, 116, 50)
PURPLE = (137, 91, 214)
GREEN = (60, 152, 73)
PINK = (230, 89, 149)
CYAN = (0, 174, 210)

BLUE_BG = (238, 246, 255)
TEAL_BG = (235, 252, 249)
ORANGE_BG = (255, 246, 238)
PURPLE_BG = (248, 242, 255)
GREEN_BG = (241, 251, 242)
PINK_BG = (255, 243, 248)
CYAN_BG = (235, 251, 255)
YELLOW_BG = (255, 250, 230)


def font(size: int, bold: bool = False, hand: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        paths = ["/System/Library/Fonts/Menlo.ttc"]
    elif hand:
        paths = [
            "/System/Library/Fonts/MarkerFelt.ttc",
            "/System/Library/Fonts/Supplemental/ChalkboardSE.ttc",
            "/System/Library/Fonts/Supplemental/Comic Sans MS Bold.ttf",
        ]
    elif bold:
        paths = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
        ]
    else:
        paths = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


F_TITLE = font(42, hand=True)
F_STAGE = font(32, hand=True)
F_H = font(23, hand=True)
F_BODY = font(19)
F_SMALL = font(16)
F_TINY = font(14)
F_MONO = font(17, mono=True)


def text_box(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    b = draw.textbbox((0, 0), text, font=fnt)
    return b[2] - b[0], b[3] - b[1]


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont, max_w: int) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        cur = ""
        for word in para.split(" "):
            cand = word if not cur else f"{cur} {word}"
            if text_box(draw, cand, fnt)[0] <= max_w:
                cur = cand
            else:
                if cur:
                    lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
    return lines


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    s: str,
    fnt: ImageFont.ImageFont,
    fill: tuple[int, int, int] = INK,
    max_w: int | None = None,
    line_h: int | None = None,
) -> int:
    x, y = xy
    if max_w is None:
        draw.text((x, y), s, font=fnt, fill=fill)
        return y + text_box(draw, s, fnt)[1]
    yy = y
    if line_h is None:
        line_h = int(fnt.size * 1.25)
    for line in wrap(draw, s, fnt, max_w):
        draw.text((x, yy), line, font=fnt, fill=fill)
        yy += line_h
    return yy


def dashed_line(
    draw: ImageDraw.ImageDraw,
    p1: tuple[float, float],
    p2: tuple[float, float],
    color: tuple[int, int, int],
    width: int = 3,
    dash: int = 16,
    gap: int = 9,
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    pos = 0.0
    while pos < length:
        end = min(pos + dash, length)
        draw.line((x1 + dx * pos, y1 + dy * pos, x1 + dx * end, y1 + dy * end), fill=color, width=width)
        pos += dash + gap


def dashed_round_rect(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    radius: int,
    color: tuple[int, int, int],
    width: int = 3,
    fill: tuple[int, int, int] | None = None,
    dash: int = 18,
    gap: int = 10,
) -> None:
    x1, y1, x2, y2 = xy
    if fill:
        draw.rounded_rectangle(xy, radius=radius, fill=fill)
    pts: list[tuple[float, float]] = []
    step = 5
    for x in range(x1 + radius, x2 - radius + 1, step):
        pts.append((x, y1))
    for a in range(-90, 1, 5):
        pts.append((x2 - radius + radius * math.cos(math.radians(a)), y1 + radius + radius * math.sin(math.radians(a))))
    for y in range(y1 + radius, y2 - radius + 1, step):
        pts.append((x2, y))
    for a in range(0, 91, 5):
        pts.append((x2 - radius + radius * math.cos(math.radians(a)), y2 - radius + radius * math.sin(math.radians(a))))
    for x in range(x2 - radius, x1 + radius - 1, -step):
        pts.append((x, y2))
    for a in range(90, 181, 5):
        pts.append((x1 + radius + radius * math.cos(math.radians(a)), y2 - radius + radius * math.sin(math.radians(a))))
    for y in range(y2 - radius, y1 + radius - 1, -step):
        pts.append((x1, y))
    for a in range(180, 271, 5):
        pts.append((x1 + radius + radius * math.cos(math.radians(a)), y1 + radius + radius * math.sin(math.radians(a))))

    period = dash + gap
    dist = 0.0
    for a, b in zip(pts, pts[1:] + pts[:1]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if int(dist) % period < dash:
            draw.line((a, b), fill=color, width=width)
        dist += seg


def arrow(draw: ImageDraw.ImageDraw, p1: tuple[int, int], p2: tuple[int, int], color: tuple[int, int, int] = INK, width: int = 5) -> None:
    draw.line((p1, p2), fill=color, width=width)
    ang = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    head = 18
    left = (p2[0] - head * math.cos(ang - math.pi / 6), p2[1] - head * math.sin(ang - math.pi / 6))
    right = (p2[0] - head * math.cos(ang + math.pi / 6), p2[1] - head * math.sin(ang + math.pi / 6))
    draw.polygon([p2, left, right], fill=color)


def routed_arrow(
    draw: ImageDraw.ImageDraw,
    pts: list[tuple[int, int]],
    color: tuple[int, int, int] = INK,
    width: int = 5,
) -> None:
    for a, b in zip(pts, pts[1:-1]):
        draw.line((a, b), fill=color, width=width)
    arrow(draw, pts[-2], pts[-1], color=color, width=width)


def panel(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    color: tuple[int, int, int],
    fill: tuple[int, int, int],
) -> None:
    dashed_round_rect(draw, xy, 22, color, width=3, fill=fill)
    x1, y1, x2, _ = xy
    tw, _ = text_box(draw, title, F_STAGE)
    draw.text((x1 + (x2 - x1 - tw) / 2, y1 + 17), title, font=F_STAGE, fill=color)


def module(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    body: str = "",
    color: tuple[int, int, int] = INK,
    fill: tuple[int, int, int] = WHITE,
    icon: str | None = None,
    tag: str | None = None,
) -> None:
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=10, fill=fill, outline=color, width=2)
    if icon:
        draw_icon(draw, icon, x1 + 16, y1 + 18, 48, color)
        tx = x1 + 78
    else:
        tx = x1 + 18
    yy = draw_text(draw, (tx, y1 + 16), title, F_H, INK, max_w=x2 - tx - 18, line_h=25) + 6
    if body:
        draw_text(draw, (tx, yy), body, F_SMALL, MUTED, max_w=x2 - tx - 18, line_h=20)
    if tag:
        tag_w, _ = text_box(draw, tag, F_TINY)
        draw.rounded_rectangle((x1 + 16, y2 - 32, x1 + 32 + tag_w, y2 - 10), radius=5, fill=WHITE, outline=color, width=1)
        draw.text((x1 + 24, y2 - 28), tag, font=F_TINY, fill=color)


def draw_icon(draw: ImageDraw.ImageDraw, kind: str, x: int, y: int, s: int, color: tuple[int, int, int]) -> None:
    if kind == "video":
        for off in (0, 8, 16):
            draw.rounded_rectangle((x + off, y + off, x + s + off, y + s * 0.7 + off), radius=5, fill=(238, 244, 255), outline=color, width=2)
        draw.polygon([(x + 30, y + 27), (x + 30, y + 55), (x + 55, y + 41)], fill=color)
    elif kind == "doc":
        draw.rectangle((x + 8, y + 3, x + s - 4, y + s + 10), fill=WHITE, outline=color, width=2)
        draw.polygon([(x + s - 18, y + 3), (x + s - 4, y + 17), (x + s - 18, y + 17)], fill=(230, 236, 246), outline=color)
        for i in range(4):
            draw.line((x + 17, y + 25 + i * 12, x + s - 15, y + 25 + i * 12), fill=color, width=2)
    elif kind == "robot":
        draw.rounded_rectangle((x + 4, y + 12, x + s, y + s), radius=10, fill=(212, 239, 255), outline=color, width=2)
        draw.line((x + s / 2, y + 12, x + s / 2, y), fill=color, width=2)
        draw.ellipse((x + s / 2 - 5, y - 7, x + s / 2 + 5, y + 3), fill=(255, 235, 120), outline=color)
        draw.ellipse((x + 18, y + 32, x + 26, y + 40), fill=color)
        draw.ellipse((x + 42, y + 32, x + 50, y + 40), fill=color)
        draw.arc((x + 22, y + 38, x + 48, y + 56), 0, 180, fill=color, width=2)
    elif kind == "target":
        for r in (26, 18, 9):
            draw.ellipse((x + s / 2 - r, y + s / 2 - r, x + s / 2 + r, y + s / 2 + r), outline=color, width=3)
        draw.line((x + s / 2, y + 4, x + s / 2, y + s - 4), fill=color, width=2)
        draw.line((x + 4, y + s / 2, x + s - 4, y + s / 2), fill=color, width=2)
    elif kind == "embed":
        pts = [(x + 12, y + 18), (x + 38, y + 10), (x + 55, y + 32), (x + 25, y + 50), (x + 52, y + 58)]
        for a, b in zip(pts, pts[1:]):
            draw.line((a, b), fill=color, width=2)
        for px, py in pts:
            draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=(245, 250, 255), outline=color, width=2)
    elif kind == "layout":
        draw.rectangle((x + 4, y + 8, x + s, y + s), fill=(246, 248, 255), outline=color, width=2)
        draw.rectangle((x + 12, y + 42, x + 46, y + 56), fill=(220, 235, 255), outline=color, width=2)
        draw.rectangle((x + 40, y + 18, x + 56, y + 38), fill=(255, 240, 210), outline=color, width=2)
        draw.rectangle((x + 14, y + 16, x + 29, y + 34), fill=(225, 250, 230), outline=color, width=2)
    elif kind == "chart":
        draw.arc((x + 4, y + 4, x + s, y + s), 20, 350, fill=color, width=6)
        draw.rectangle((x + 12, y + 45, x + 22, y + 64), fill=color)
        draw.rectangle((x + 30, y + 34, x + 40, y + 64), fill=color)
        draw.rectangle((x + 48, y + 20, x + 58, y + 64), fill=color)
    elif kind == "warn":
        draw.polygon([(x + s / 2, y + 5), (x + s - 4, y + s), (x + 4, y + s)], fill=(255, 242, 210), outline=color)
        draw.line((x + s / 2, y + 24, x + s / 2, y + 48), fill=color, width=4)
        draw.ellipse((x + s / 2 - 3, y + 56, x + s / 2 + 3, y + 62), fill=color)
    else:
        draw.ellipse((x + 6, y + 6, x + s, y + s), fill=(245, 249, 255), outline=color, width=2)


def title(draw: ImageDraw.ImageDraw, s: str) -> None:
    tw, _ = text_box(draw, s, F_TITLE)
    draw.text(((W - tw) / 2, 26), s, font=F_TITLE, fill=INK)


def variant_1() -> Image.Image:
    img = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(img)
    title(d, "Scene Continuity Evaluation Suite")

    panel(d, (90, 90, 2310, 320), "1. Inputs & Evidence Construction", GRAY, (252, 252, 252))
    module(d, (150, 150, 430, 275), "Generated Video", "multi-shot episode V", BLUE, BLUE_BG, "video")
    module(d, (520, 150, 800, 275), "Shot Captions", "shot-level prompt C", PURPLE, PURPLE_BG, "doc")
    module(d, (900, 150, 1240, 275), "Shot Alignment", "boundaries + keyframes K", GREEN, GREEN_BG, "target")
    module(d, (1360, 150, 1710, 275), "Prompt Entity List", "characters, objects,\nactions, relations", PURPLE, PURPLE_BG, "doc", "LLM/rules")
    module(d, (1830, 150, 2210, 275), "Intrinsic Evidence Pool", "global state, salient objects,\nlayout anchors", TEAL, TEAL_BG, "layout")
    for a, b in [((430, 212), (520, 212)), ((800, 212), (900, 212)), ((1240, 212), (1360, 212))]:
        arrow(d, a, b)
    routed_arrow(d, [(1240, 250), (1300, 250), (1300, 298), (1785, 298), (1785, 240), (1830, 212)])

    panel(d, (90, 365, 2310, 835), "2. Local Model Calls & Verified Opportunities", BLUE, (248, 251, 255))
    draw_text(d, (145, 418), "(A) Prompt-grounded branch", F_H, BLUE)
    module(d, (150, 470, 465, 620), "Ground Prompt Entities", "detect scheduled entities;\nCLIP gate gives present / weak / absent", TEAL, TEAL_BG, "target", "OVDet+CLIP")
    module(d, (555, 470, 870, 620), "Judge Fidelity", "crop/action must match caption\nbefore cross-shot scoring", ORANGE, ORANGE_BG, "robot", "MLLM JSON")
    module(d, (960, 470, 1275, 620), "Compare Recurring Entities", "identity, appearance,\nstate, relation", BLUE, BLUE_BG, "embed", "DINOv2/Face")
    module(d, (1365, 470, 1680, 620), "PG Opportunity Set", "scheduled slots + verified\ncross-shot pairs", GREEN, GREEN_BG, "chart")
    for x in [465, 870, 1275]:
        arrow(d, (x, 545), (x + 90, 545), BLUE)

    draw_text(d, (145, 660), "(B) Intrinsic self-consistency branch", F_H, TEAL)
    module(d, (150, 705, 465, 795), "Group Comparable Views", "background embedding + geometry", GREEN, GREEN_BG, "layout", "DINOv2+Geom.")
    module(d, (555, 705, 870, 795), "Mine Scene Evidence", "global, salient objects, layout graph", TEAL, TEAL_BG, "target", "OVDet")
    module(d, (960, 705, 1275, 795), "Checkability Judge", "ignore normal angle / occlusion", ORANGE, ORANGE_BG, "robot", "MLLM JSON")
    module(d, (1365, 705, 1680, 795), "IS Opportunity Set", "comparable groups + visible evidence", GREEN, GREEN_BG, "chart")
    for x in [465, 870, 1275]:
        arrow(d, (x, 750), (x + 90, 750), TEAL)

    module(d, (1810, 510, 2200, 740), "Structured Evidence Store", "crops, bboxes, embeddings,\nlayout relations, MLLM JSON,\nconfidence + checkability flags", PINK, PINK_BG, "doc")
    arrow(d, (1680, 545), (1810, 600), PINK)
    arrow(d, (1680, 750), (1810, 650), PINK)

    panel(d, (90, 880, 2310, 1250), "3. Deterministic Metrics & Typed Findings", ORANGE, (255, 250, 246))
    module(d, (150, 960, 465, 1115), "Coverage Metrics", "PG-Coverage\nIS-Coverage / Richness", BLUE, BLUE_BG, "chart")
    module(d, (555, 960, 870, 1115), "Consistency Metrics", "PG-Consistency\nIS-Consistency", TEAL, TEAL_BG, "chart")
    module(d, (960, 960, 1330, 1115), "Scene Continuity Score", "SCS = sum w_c * score_c * opp_c\n      / sum w_c * opp_c", ORANGE, YELLOW_BG, "target", "Formula")
    module(d, (1430, 960, 1805, 1115), "Typed Findings", "Missing, Appearance Drift,\nState Drift, Spatial Drift,\nLighting Drift", PINK, PINK_BG, "warn")
    module(d, (1900, 960, 2220, 1115), "Audit Artifacts", "keyframes, crops,\nview groups, JSON logs", PURPLE, PURPLE_BG, "doc")
    for x in [465, 870, 1330, 1805]:
        arrow(d, (x, 1038), (x + 90, 1038), ORANGE)
    return img


def variant_2() -> Image.Image:
    img = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(img)
    title(d, "Algorithmic Evaluation Overview")

    panel(d, (50, 95, 720, 420), "1. Episode Preparation", GREEN, GREEN_BG)
    module(d, (90, 165, 290, 285), "Video", "generated episode", BLUE, BLUE_BG, "video")
    module(d, (360, 165, 560, 285), "Caption", "shot prompts", PURPLE, PURPLE_BG, "doc")
    arrow(d, (290, 225), (360, 225))
    module(d, (130, 315, 640, 385), "Align shots, sample keyframes, produce foreground/background masks", "", GREEN, WHITE, "target")

    panel(d, (760, 95, 1500, 420), "2. Prompt-Grounded Continuity", BLUE, BLUE_BG)
    module(d, (800, 165, 1030, 285), "Parse", "entity schedule", PURPLE, PURPLE_BG, "doc", "LLM")
    module(d, (1085, 165, 1315, 285), "Ground", "crop + CLIP gate", TEAL, TEAL_BG, "target", "OVDet")
    arrow(d, (1030, 225), (1085, 225), BLUE)
    module(d, (800, 315, 1030, 385), "Fidelity", "caption match", ORANGE, ORANGE_BG, "robot", "MLLM")
    module(d, (1085, 315, 1315, 385), "Consistency", "ID / state / relation", BLUE, WHITE, "embed", "DINOv2")
    arrow(d, (1030, 350), (1085, 350), BLUE)

    panel(d, (1540, 95, 2350, 420), "3. Intrinsic Scene Consistency", TEAL, TEAL_BG)
    module(d, (1580, 165, 1810, 285), "View Grouping", "comparable shots", GREEN, GREEN_BG, "layout", "DINO+Geom.")
    module(d, (1870, 165, 2100, 285), "Evidence", "global / object / layout", TEAL, WHITE, "target", "OVDet")
    arrow(d, (1810, 225), (1870, 225), TEAL)
    module(d, (1580, 315, 1810, 385), "Checkability", "visible? comparable?", ORANGE, ORANGE_BG, "robot", "MLLM")
    module(d, (1870, 315, 2100, 385), "Compare", "state / object / graph", TEAL, WHITE, "embed")
    arrow(d, (1810, 350), (1870, 350), TEAL)

    panel(d, (50, 460, 1500, 900), "4. Metrics Produced by Fixed Aggregation", ORANGE, ORANGE_BG)
    module(d, (110, 540, 410, 695), "PG-Coverage", "# present scheduled slots\n/ # scheduled slots", BLUE, WHITE, "chart")
    module(d, (465, 540, 765, 695), "PG-Consistency", "stable prompt entities\namong verified pairs", BLUE, WHITE, "chart")
    module(d, (820, 540, 1120, 695), "IS-Coverage", "verifiable generated\nscene evidence", TEAL, WHITE, "chart")
    module(d, (1175, 540, 1450, 695), "IS-Consistency", "stable intrinsic evidence\nwithin view groups", TEAL, WHITE, "chart")
    module(d, (380, 740, 1170, 850), "SCS = opportunity-weighted aggregate", "sum_c w_c * score_c * opp_c / sum_c w_c * opp_c", ORANGE, YELLOW_BG, "target", "Formula")
    for x in [410, 765, 1120]:
        arrow(d, (x, 620), (x + 55, 620), ORANGE)

    panel(d, (1540, 460, 2350, 900), "5. Diagnostic Judge Output", PINK, PINK_BG)
    module(d, (1600, 540, 1830, 680), "Finding Type", "Missing\nAppearance Drift\nState Drift", PINK, WHITE, "warn")
    module(d, (1900, 540, 2130, 680), "Localization", "element + affected shots\n+ confidence", PURPLE, WHITE, "target")
    module(d, (1700, 730, 2210, 850), "Audit Trace", "keyframes, crops, view groups, evidence scores, MLLM JSON", GREEN, WHITE, "doc")

    panel(d, (50, 945, 2350, 1260), "6. Evaluator Validation by Controlled Perturbations", CYAN, CYAN_BG)
    module(d, (120, 1030, 390, 1160), "Known Stable Clip", "clean reference generation", GREEN, WHITE, "video")
    module(d, (500, 1030, 840, 1160), "Inject Continuity Errors", "delete object, recolor item,\nmove layout anchor, relight scene", ORANGE, WHITE, "warn")
    module(d, (960, 1030, 1300, 1160), "Run Evaluator", "expect score monotonicity\nand correct finding type", BLUE, WHITE, "robot")
    module(d, (1420, 1030, 1780, 1160), "Report Reliability", "detection rate, false positive,\nlocalization accuracy", CYAN, WHITE, "chart")
    arrow(d, (390, 1095), (500, 1095), CYAN)
    arrow(d, (840, 1095), (960, 1095), CYAN)
    arrow(d, (1300, 1095), (1420, 1095), CYAN)
    return img


def variant_3() -> Image.Image:
    img = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(img)
    title(d, "Two-Track Continuity Scoring")

    panel(d, (70, 95, 700, 1160), "1. Shared Inputs", GREEN, GREEN_BG)
    module(d, (130, 190, 330, 320), "Video V", "multi-shot generation", BLUE, WHITE, "video")
    module(d, (420, 190, 620, 320), "Captions C", "shot prompts", PURPLE, WHITE, "doc")
    arrow(d, (330, 255), (420, 255))
    module(d, (125, 410, 640, 560), "Shot Alignment", "prompt-anchored boundaries\n+ keyframe sampling", GREEN, WHITE, "target")
    module(d, (125, 650, 640, 830), "Evidence Cache", "frames, masks, crops,\nembeddings, layout anchors", TEAL, WHITE, "layout")
    arrow(d, (380, 320), (380, 410), GREEN)
    arrow(d, (380, 560), (380, 650), GREEN)

    panel(d, (760, 95, 1515, 560), "2. Prompt-Grounded Track", BLUE, BLUE_BG)
    module(d, (805, 190, 1045, 325), "Targets", "caption-mentioned\nentities / states", PURPLE, WHITE, "doc", "LLM")
    module(d, (1095, 190, 1345, 325), "Visual Grounding", "bbox + crop + gate", TEAL, WHITE, "target", "OVDet+CLIP")
    arrow(d, (1045, 255), (1095, 255), BLUE)
    module(d, (805, 380, 1045, 505), "Semantic Gate", "fidelity + action", ORANGE, WHITE, "robot", "MLLM")
    module(d, (1095, 380, 1345, 505), "Consistency", "appearance / ID /\nstate / relation", BLUE, WHITE, "embed", "DINOv2")
    arrow(d, (1045, 442), (1095, 442), BLUE)

    panel(d, (760, 620, 1515, 1160), "3. Intrinsic Track", TEAL, TEAL_BG)
    module(d, (805, 715, 1045, 850), "View Groups", "only comparable views", GREEN, WHITE, "layout", "DINO+Geom.")
    module(d, (1095, 715, 1345, 850), "Scene Evidence", "global, object, layout", TEAL, WHITE, "target", "OVDet")
    arrow(d, (1045, 780), (1095, 780), TEAL)
    module(d, (805, 920, 1045, 1045), "Checkability", "partial view aware", ORANGE, WHITE, "robot", "MLLM")
    module(d, (1095, 920, 1345, 1045), "Comparison", "state / object /\nrelation graph", TEAL, WHITE, "embed")
    arrow(d, (1045, 982), (1095, 982), TEAL)

    panel(d, (1575, 95, 2325, 725), "4. Metric Vector", ORANGE, ORANGE_BG)
    metrics = [
        ("PG-Coverage", BLUE),
        ("PG-Consistency", BLUE),
        ("IS-Coverage / Richness", TEAL),
        ("IS-Consistency", TEAL),
        ("SCS", ORANGE),
    ]
    y = 190
    for name, color in metrics:
        module(d, (1640, y, 2210, y + 80), name, "computed over verified opportunities", color, WHITE, "chart")
        y += 95
    module(d, (1640, 668, 2210, 718), "coverage is reported with consistency", "", PINK, PINK_BG, "warn")
    arrow(d, (1515, 325), (1575, 280), BLUE)
    arrow(d, (1515, 890), (1575, 520), TEAL)

    panel(d, (1575, 780, 2325, 1160), "5. Findings & Audit", PINK, PINK_BG)
    module(d, (1640, 870, 1880, 1025), "Typed Error", "Missing\nAppearance Drift\nSpatial Drift", PINK, WHITE, "warn")
    module(d, (1960, 870, 2200, 1025), "Evidence", "affected shots\nscores\nconfidence", GREEN, WHITE, "doc")
    arrow(d, (1880, 947), (1960, 947), PINK)
    return img


def variant_4() -> Image.Image:
    img = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(img)
    title(d, "Where Models Enter the Evaluation Suite")

    panel(d, (70, 100, 2330, 360), "1. Construct Checkable Evidence", GRAY, (252, 252, 252))
    module(d, (140, 175, 370, 300), "Captions", "prompt entities", PURPLE, PURPLE_BG, "doc", "LLM")
    module(d, (480, 175, 710, 300), "Frames", "keyframes + masks", BLUE, BLUE_BG, "video")
    module(d, (820, 175, 1110, 300), "Grounding", "scheduled entities + salient objects", TEAL, TEAL_BG, "target", "OVDet+CLIP")
    module(d, (1240, 175, 1530, 300), "Embedding", "crop / frame / face features", BLUE, WHITE, "embed", "DINOv2/Face")
    module(d, (1660, 175, 1970, 300), "View Grouping", "comparable-shot clusters", GREEN, GREEN_BG, "layout", "Geometry")
    arrow(d, (370, 238), (480, 238))
    arrow(d, (710, 238), (820, 238))
    arrow(d, (1110, 238), (1240, 238))
    arrow(d, (1530, 238), (1660, 238))

    panel(d, (70, 430, 1125, 920), "2. MLLM as Local Semantic Judge", ORANGE, ORANGE_BG)
    module(d, (130, 520, 440, 680), "Prompt Fidelity", "Does this crop match\nthe caption entity?", ORANGE, WHITE, "robot", "MLLM JSON")
    module(d, (520, 520, 830, 680), "Action / State", "Is the instructed action\nvisible across frames?", ORANGE, WHITE, "robot", "MLLM JSON")
    module(d, (320, 735, 630, 865), "Checkability", "Can these views be\nfairly compared?", ORANGE, WHITE, "robot", "MLLM JSON")
    arrow(d, (440, 600), (520, 600), ORANGE)
    arrow(d, (675, 680), (575, 735), ORANGE)

    panel(d, (1195, 430, 2330, 920), "3. Non-MLLM Computable Scores", BLUE, BLUE_BG)
    module(d, (1260, 520, 1540, 680), "Presence / Coverage", "scheduled slots,\nverified evidence count", BLUE, WHITE, "chart")
    module(d, (1625, 520, 1905, 680), "Embedding Similarity", "identity and appearance\ncentroid similarity", BLUE, WHITE, "embed")
    module(d, (1990, 520, 2265, 680), "Layout Agreement", "relation graph\nconsistency", TEAL, WHITE, "layout")
    module(d, (1450, 745, 2070, 865), "Opportunity-Weighted Aggregation", "SCS = sum w_c * score_c * opp_c / sum w_c * opp_c", ORANGE, YELLOW_BG, "target", "Formula")
    for x in [1540, 1905]:
        arrow(d, (x, 600), (x + 85, 600), BLUE)
    arrow(d, (2128, 680), (1880, 745), BLUE)

    panel(d, (70, 980, 2330, 1250), "4. Final Outputs", PINK, PINK_BG)
    module(d, (150, 1055, 480, 1185), "Leaderboard Metrics", "PG-Cov, PG-Con,\nIS-Cov, IS-Con, SCS", ORANGE, WHITE, "chart")
    module(d, (610, 1055, 940, 1185), "Typed Findings", "element, error type,\naffected shots, severity", PINK, WHITE, "warn")
    module(d, (1070, 1055, 1400, 1185), "Audit Artifacts", "crops, view groups,\nMLLM JSON, scores", GREEN, WHITE, "doc")
    module(d, (1530, 1055, 2020, 1185), "Perturbation Validation", "controlled edits test monotonicity,\nfalse positives, localization", CYAN, WHITE, "target")
    for x in [480, 940, 1400]:
        arrow(d, (x, 1120), (x + 130, 1120), PINK)
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    variants = [variant_1(), variant_2(), variant_3(), variant_4()]
    for i, img in enumerate(variants, 1):
        path = OUT / f"evaluation_suite_refstyle_variant_{i}.png"
        img.save(path, quality=96)
        print(path)


if __name__ == "__main__":
    main()
