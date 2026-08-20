"""Formal 3-panel schematic: large / small / ghost attention (no decorative style)."""
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 720
BG = (255, 255, 255)
INK = (28, 28, 28)
MUTED = (90, 90, 90)
RULE = (40, 40, 40)
PANEL = (255, 255, 255)
# Single-hue navy ramp (paper figure, not a rainbow heatmap)
C0 = (248, 249, 250)
C1 = (27, 54, 93)

N = 20  # patch grid
MARGIN_X = 80
TOP = 36
GAP = 56
LABEL_H = 124
PANEL_W = (W - 2 * MARGIN_X - 2 * GAP) // 3
PANEL_H = H - TOP - LABEL_H - 28


def lerp(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def gauss(x, y, cx, cy, sx, sy):
    return math.exp(-((x - cx) ** 2) / (2 * sx * sx) - ((y - cy) ** 2) / (2 * sy * sy))


def plateau(i, j, cx, cy, rx, ry, edge=0.55):
    """Filled ellipse with a short falloff — high purity, not a soft blob."""
    d = math.sqrt(((i - cx) / rx) ** 2 + ((j - cy) / ry) ** 2)
    if d <= 1.0 - edge / max(rx, ry):
        return 1.0
    if d >= 1.0:
        return 0.0
    t = (1.0 - d) / (edge / max(rx, ry))
    return t * t * (3.0 - 2.0 * t)


def field_large(i, j):
    return 0.02 + 0.96 * plateau(i, j, 9.6, 10.1, 6.2, 5.4, edge=0.9)


def field_small(i, j):
    return 0.02 + 0.96 * plateau(i, j, 7.2, 7.6, 1.55, 1.35, edge=0.45)


def field_ghost(i, j):
    # Always-second: low, nearly uniform, no compact support.
    v = (
        0.40 * gauss(i, j, 5.5, 6.5, 8.0, 7.0)
        + 0.35 * gauss(i, j, 14.5, 13.5, 7.5, 6.5)
        + 0.25 * gauss(i, j, 10.0, 4.0, 7.0, 6.0)
    )
    return 0.11 + 0.10 * v


def render_grid(fn):
    vals = [[fn(i + 0.5, j + 0.5) for i in range(N)] for j in range(N)]
    lo, hi = min(min(r) for r in vals), max(max(r) for r in vals)
    # Shared visual scale 0..1 so panel (c) stays visibly faint.
    scale_hi = 1.0
    img = Image.new("RGB", (PANEL_W, PANEL_H), PANEL)
    draw = ImageDraw.Draw(img)
    # Inner square plot area
    pad = 18
    side = min(PANEL_W, PANEL_H) - 2 * pad
    x0 = (PANEL_W - side) // 2
    y0 = (PANEL_H - side) // 2
    cell = side / N
    for j in range(N):
        for i in range(N):
            t = max(0.0, min(1.0, vals[j][i] / scale_hi))
            # Slight gamma so mid values stay readable without looking "glowy"
            t = t ** 0.92
            color = lerp(C0, C1, t)
            x1 = x0 + int(round(i * cell))
            y1 = y0 + int(round(j * cell))
            x2 = x0 + int(round((i + 1) * cell))
            y2 = y0 + int(round((j + 1) * cell))
            draw.rectangle([x1, y1, x2 - 1, y2 - 1], fill=color)
    # Grid every 4 cells (token blocks), hairline
    for k in range(N + 1):
        p = x0 + int(round(k * cell))
        q = y0 + int(round(k * cell))
        wline = 1 if k % 4 else 1
        col = (186, 190, 196) if k % 4 else (168, 172, 178)
        draw.line([(p, y0), (p, y0 + side)], fill=col, width=wline)
        draw.line([(x0, q), (x0 + side, q)], fill=col, width=wline)
    draw.rectangle([x0, y0, x0 + side, y0 + side], outline=RULE, width=2)
    return img, lo, hi


def load_font(path, size):
    return ImageFont.truetype(path, size)


def main():
    out = Path("/mnt/ssd2/hmlee/SlotCurri/event_analysis/slide_purity_three_cases.png")
    font_r = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_b = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    panel_f = load_font(font_b, 32)
    math_f = load_font(font_r, 30)
    note_f = load_font(font_r, 20)

    im = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(im)

    panels = [
        ("(a)  Large object", "m  high", "c  ≈  1", field_large),
        ("(b)  Small object", "m  low", "c  ≈  1", field_small),
        ("(c)  Ghost  (always-second)", "m  low", "c  low", field_ghost),
    ]

    for k, (name, mlab, clab, fn) in enumerate(panels):
        x = MARGIN_X + k * (PANEL_W + GAP)
        y = TOP
        grid, _, _ = render_grid(fn)
        im.paste(grid, (x, y))
        dr.rectangle([x, y, x + PANEL_W - 1, y + PANEL_H - 1], outline=RULE, width=2)

        # Caption block
        cy = y + PANEL_H + 18
        nw = dr.textlength(name, font=panel_f)
        dr.text((x + (PANEL_W - nw) / 2, cy), name, font=panel_f, fill=INK)

        # Two stats on one baseline, aligned as a pair
        gap_stats = 36
        mw = dr.textlength(mlab, font=math_f)
        cw = dr.textlength(clab, font=math_f)
        total = mw + gap_stats + cw
        sx = x + (PANEL_W - total) / 2
        sy = cy + 42
        mcol = INK
        ccol = INK if "≈" in clab else MUTED
        # ghost c is the distinguishing low value — keep it ink, not muted
        ccol = INK
        dr.text((sx, sy), mlab, font=math_f, fill=mcol)
        dr.text((sx + mw + gap_stats, sy), clab, font=math_f, fill=ccol)

    legend = "One slot’s attention over patches.  Darker = higher attention."
    lw = dr.textlength(legend, font=note_f)
    dr.text(((W - lw) / 2, H - 32), legend, font=note_f, fill=MUTED)

    im.save(out, "PNG")
    print(f"wrote {out}  {im.size}")


if __name__ == "__main__":
    main()
