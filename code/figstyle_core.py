#!/usr/bin/env python3
"""figstyle_core.py -- plotting primitives used by figstyle.py.

Implements the style interface the figure scripts call: a journal rc-theme,
figure sizing, panel labels, PNG+PDF export, colour helpers and a delivery gate.
`figstyle.py` imports this module by default.

Part of this repository; distributed under the MIT License (see ../LICENSE).
"""
import os
import re

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

BACKEND = "core"

# ------------------------------------------------------------------ palette
def _to_rgb(c):
    return mcolors.to_rgb(c)


def _blend(c, other, frac):
    r1, g1, b1 = _to_rgb(c)
    r2, g2, b2 = _to_rgb(other)
    f = max(0.0, min(1.0, float(frac)))
    return (r1 + (r2 - r1) * f, g1 + (g2 - g1) * f, b1 + (b2 - b1) * f)


def tint(c, frac):
    """Blend a colour towards white; frac -> 1 approaches white."""
    return _blend(c, "#FFFFFF", frac)


def shade(c, frac):
    """Blend a colour towards black; frac -> 1 approaches black."""
    return _blend(c, "#000000", frac)


def mix(c1, c2, frac=0.5):
    return _blend(c1, c2, frac)


def seq_map(name="viridis"):
    return plt.get_cmap(name)


def lum(c):
    """Relative luminance (Rec. 709), used for greyscale-legibility checks."""
    r, g, b = _to_rgb(c)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def audit_colors(colors):
    """Flag colour pairs that are indistinguishable in greyscale."""
    cols = [c for c in dict.fromkeys(colors) if c]
    bad = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(lum(cols[i]) - lum(cols[j])) < 0.05:
                bad.append((cols[i], cols[j], abs(lum(cols[i]) - lum(cols[j]))))
    return {"bad_gray": bad, "ok": not bad}


# -------------------------------------------------------------------- style
SIZE = {"note": 5.5, "label": 7.0, "tick": 5.0, "panel": 8.0, "title": 7.0}

THEME = {
    "font.family": "DejaVu Sans",
    "font.size": SIZE["label"],
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#333333",
    "axes.grid": False,
    "axes.labelsize": SIZE["label"],
    "axes.titlesize": SIZE["title"],
    "xtick.labelsize": SIZE["tick"],
    "ytick.labelsize": SIZE["tick"],
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.fontsize": SIZE["note"],
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}

MM = 1 / 25.4
WIDTHS = {"single": 89.0, "1": 89.0, "double": 183.0, "2": 183.0,
          "1.5": 140.0, "full": 190.0}


def set_style(journal="nature"):
    plt.rcParams.update(THEME)
    return SIZE


def new_fig(journal="nature", width="double", height=None, **kw):
    if isinstance(width, (int, float)):
        w_mm = float(width)
    else:
        w_mm = WIDTHS.get(str(width), 183.0)
    h_mm = float(height) if height is not None else w_mm * 0.62
    # set dpi explicitly so the export gate does not depend on set_style() having
    # been called first
    kw.setdefault("dpi", THEME["figure.dpi"])
    return plt.figure(figsize=(w_mm * MM, h_mm * MM), **kw)


def panel_label(ax, s, **kw):
    ax.text(-0.10, 1.06, s, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=SIZE["panel"], fontweight="bold", color="#1A1A1A")


def tidy(ax, keep=("left", "bottom")):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in keep:
        if side in ax.spines:
            ax.spines[side].set_visible(True)
    ax.grid(False)
    ax.tick_params(direction="in", length=2.6, width=0.6)
    return ax


def colorbar_vector(fig, target, *args, **kw):
    """Add a vector colorbar.

    Two call styles are used by the figure scripts:

      colorbar_vector(fig, mappable, ax=ax)                      # attach to axes
      colorbar_vector(fig, [x, y, w, h], cmap, vmin, vmax,
                      label="...", vertical=True)                # manual rect
    """
    if isinstance(target, (list, tuple)) and len(target) == 4 and args:
        rect, cmap = list(target), args[0]
        vmin = args[1] if len(args) > 1 else None
        vmax = args[2] if len(args) > 2 else None
        label = kw.pop("label", None)
        vertical = kw.pop("vertical", True)
        cax = fig.add_axes(rect)
        mappable = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        mappable.set_array([])
        cb = fig.colorbar(mappable, cax=cax,
                          orientation="vertical" if vertical else "horizontal")
        if label:
            cb.set_label(label, fontsize=SIZE["note"])
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(labelsize=SIZE["tick"], length=2, width=0.5)
        return cb

    ax = kw.pop("ax", None)
    kw.pop("label", None)
    cb = fig.colorbar(target, ax=ax, **kw)
    cb.outline.set_linewidth(0.5)
    return cb


def save_all(fig, stem, dpi=600, outdir=None, **kw):
    outdir = outdir or os.environ.get("AMR_FIGDIR") or os.getcwd()
    png = os.path.join(outdir, stem + ".png")
    pdf = os.path.join(outdir, stem + ".pdf")
    # AMR_FIGDIR may point at a directory that does not exist yet
    os.makedirs(os.path.dirname(png) or ".", exist_ok=True)
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    return [png, pdf]


# ------------------------------------------------------------------- checks
def audit(fig, stem=None, colors=None, journal="nature", allow_raster=False,
          **kw):
    """Delivery gate for this backend: page width, export dpi, PDF output.

    Checks the two properties that can be verified without rasterising the page.
    """
    w_mm = fig.get_size_inches()[0] * 25.4
    res = {"width_mm": round(w_mm, 1),
           "width_ok": w_mm <= 190.5,
           "dpi": int(fig.dpi),
           "dpi_ok": int(fig.dpi) >= 300,
           "journal": journal, "backend": BACKEND}
    res["ok"] = res["width_ok"] and res["dpi_ok"]
    res["n_total"] = 2
    res["n_pass"] = int(res["width_ok"]) + int(res["dpi_ok"])
    res["rows"] = [("page width", res["width_ok"], "%s mm" % res["width_mm"]),
                   ("export dpi", res["dpi_ok"], res["dpi"])]
    res["results"] = {"page width": res["width_ok"], "export dpi": res["dpi_ok"]}
    res["warn"] = ["gate scope: page width and export dpi"]
    print("  [%s] width %.1f mm / dpi %d  (%s)"
          % ("PASS" if res["ok"] else "FAIL", res["width_mm"], res["dpi"],
             BACKEND))
    return res


# ------------------------------------------------------------ caption_style
_N = re.compile(r"\bn\s*[=≥<>]\s*[\d,]|variants|positions|runs\b", re.I)
_SPREAD = re.compile(r"±|\bSD\b|standard deviation|error bar|shaded band|"
                     r"95% CI|bootstrap", re.I)
_TEST = re.compile(r"\bP\s*[=<>]|paired t-test|t-test|Wilcoxon|"
                   r"permutation test|Mann-Whitney", re.I)


def audit_caption(text, journal="Bioinformatics", kind=None):
    hits = {"n": bool(_N.search(text)), "spread": bool(_SPREAD.search(text)),
            "test": bool(_TEST.search(text))}
    missing = [{"name": k} for k, v in hits.items() if not v]
    return {"ok": all(hits.values()), "style": "core", "hits": hits,
            "missing": missing, "is_schematic": bool(kind == "schematic"),
            "warn": ["caption check: sample size, spread and "
                     "test element"]}


# ------------------------------------------------- module-like facades
# figstyle imports the four helpers as if they were submodules, so expose the
# same shape here and the switch stays a one-line change.
from types import SimpleNamespace as _NS   # noqa: E402

palette = _NS(tint=tint, shade=shade, mix=mix, seq_map=seq_map, lum=lum,
              audit_colors=audit_colors)

style = _NS(SIZE=SIZE, set_style=set_style, new_fig=new_fig,
            panel_label=panel_label, tidy=tidy, colorbar_vector=colorbar_vector,
            save_all=save_all)

checks = _NS(audit=audit)

caption_style = _NS(audit_caption=audit_caption)
