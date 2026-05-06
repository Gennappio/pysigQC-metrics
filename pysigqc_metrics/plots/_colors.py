"""Maximally-distinguishable color palette for dataset identification.

Same algorithm as sigQC-master/R/dataset_colors.R — anchors the first three
colors at red, blue, green (ColorBrewer Set1) and places additional colors at
HCL hues that maximize minimum angular distance from all existing hues.
"""

from __future__ import annotations

import colorsys

ANCHOR_HEX = ("#E41A1C", "#377EB8", "#4DAF4A")  # red, blue, green
ANCHOR_HUES = (12.0, 255.0, 135.0)               # HCL hue degrees

# Shared constants for other color schemes (not dataset identity)
HEATMAP_CMAP = "RdBu_r"           # diverging blue-white-red
JET_CMAP = "jet"                   # continuous density coloring
SIG_GENE_COLOR = "#E41A1C"        # red — signature gene highlights
BACKGROUND_COLOR = "#BFBFBF"      # grey — all-gene background
QUANTILE_COLORS = {
    "outer": "#B0E0E6",            # cadetblue1 — 10%/90%
    "inner": "#1E90FF",            # dodgerblue1 — 25%/75%
    "median": "#00008B",           # darkblue — 50%
}


def _hcl_to_hex(h: float, c: float, l: float) -> str:
    """Convert HCL (hue 0-360, chroma 0-100+, luminance 0-100) to hex.

    Uses the CIE-LCh(uv) -> CIELUV -> XYZ -> sRGB pipeline, matching R's
    ``grDevices::hcl()`` output.
    """
    import math

    # LCh(uv) -> CIELUV
    h_rad = math.radians(h)
    L = l
    u = c * math.cos(h_rad)
    v = c * math.sin(h_rad)

    # CIELUV -> XYZ (D65 illuminant)
    # Reference white D65
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    un = 4 * Xn / (Xn + 15 * Yn + 3 * Zn)
    vn = 9 * Yn / (Xn + 15 * Yn + 3 * Zn)

    if L <= 0 and u == 0 and v == 0:
        return "#000000"

    if L > 7.999625:
        Y = Yn * ((L + 16) / 116) ** 3
    else:
        Y = Yn * L / 903.3

    u_prime = u / (13 * L) + un if L != 0 else 0
    v_prime = v / (13 * L) + vn if L != 0 else 0

    if v_prime == 0:
        return "#000000"

    X = Y * 9 * u_prime / (4 * v_prime)
    Z = Y * (12 - 3 * u_prime - 20 * v_prime) / (4 * v_prime)

    # XYZ -> linear sRGB
    r_lin = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    g_lin = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    b_lin = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z

    # Linear sRGB -> sRGB gamma
    def gamma(v: float) -> float:
        if v <= 0.0031308:
            return max(0.0, min(1.0, 12.92 * v))
        return max(0.0, min(1.0, 1.055 * v ** (1 / 2.4) - 0.055))

    r, g, b = gamma(r_lin), gamma(g_lin), gamma(b_lin)
    return f"#{int(r * 255 + 0.5):02X}{int(g * 255 + 0.5):02X}{int(b * 255 + 0.5):02X}"


def dataset_colors(n: int) -> list[str]:
    """Generate *n* maximally-distinguishable hex colors.

    First three are anchored at red, blue, green.
    Additional colors are placed at HCL hues that maximize minimum angular
    distance from all previously placed hues (C=80, L=60).
    """
    if n <= 0:
        return []
    if n <= 3:
        return list(ANCHOR_HEX[:n])

    placed = list(ANCHOR_HUES)
    for _ in range(n - 3):
        best_hue = 0
        best_dist = -1.0
        for cand in range(360):
            min_d = min(min(abs(cand - p), 360 - abs(cand - p)) for p in placed)
            if min_d > best_dist:
                best_dist = min_d
                best_hue = cand
        placed.append(float(best_hue))

    extra = [_hcl_to_hex(h, 80, 60) for h in placed[3:]]
    return list(ANCHOR_HEX) + extra


# Linestyle cycle for signatures (matches R's lty = 1, 2, 3, ...)
SIG_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]


def sig_linestyle(index: int) -> str | tuple:
    """Return a matplotlib linestyle for the *index*-th signature (0-based)."""
    return SIG_LINESTYLES[index % len(SIG_LINESTYLES)]
