#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
47_render_pymol_tem1.py — Figure 1(e) 的 TEM-1 结构渲染（PyMOL → 位图）

做两件事：
  1) 用 PyMOL ray-trace 渲染 TEM-1 β-内酰胺酶的 AlphaFold 模型两张图（**同一相机**）：
       A. 按「模型预测的逐残基有害性」着色（cividis 顺序色标）
       B. 按「Cβ 接触数（埋藏度）」着色（同一色标，不同量纲）
     催化 Ser70 用朱红球体标出，Ω-loop（163–178）用紫红高亮。
     N 端信号肽（1–23，AlphaFold 里是一条无序长尾，且不在 DMS 测序范围内）不渲染。
  2) 再渲一张「探针图」：只显示 Ser70（纯红）与 Ω-loop（纯绿），用来反解它们在成像上的
     像素位置 —— 这样标注就能由 matplotlib 用矢量文字叠上去，
     **PyMOL 内不写任何文字**（否则 PDF 里嵌不进字体）。

⚠️ 必须在 pymol4 环境跑： C:\\ProgramData\\anaconda3\\envs\\pymol4\\python.exe
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(REPO, "data")
RES = os.path.join(REPO, "results")

PDB = os.path.join(DATA, "AF-P62593-F1-model.pdb")
OUTDIR = os.path.join(RES, "pymol")
os.makedirs(OUTDIR, exist_ok=True)

CATALYTIC = [70]                      # 论文使用的催化 Ser70 锚点
OMEGA = list(range(163, 179))          # Ω-loop 163–178
NTERM = 23                             # N 端信号肽（不渲染）
N_BINS = 24
SIZE = 1200                            # 渲染边长（px）

S70_RGB = [0.84, 0.37, 0.0]            # #D55E00 朱红：催化 Ser70
OMEGA_RGB = [0.80, 0.47, 0.65]         # #CC79A7 紫红：Ω-loop

pp = pd.read_csv(os.path.join(RES, "tem1_per_position_pred_vs_true.csv")).set_index("pos")
sf = pd.read_csv(os.path.join(DATA, "tem1_structural_features.csv")).set_index("pos")
pos_all = list(sf.index)

del_val = {p: (float(pp.loc[p, "pred_mean"]) if p in pp.index else np.nan)
           for p in pos_all}
bur_val = {p: float(sf.loc[p, "contact_num"]) for p in pos_all}


def _bins(values, vmin, vmax):
    """把残基按数值分箱，返回 {bin_index: [positions]} 与每箱的 RGB。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    cmap = plt.get_cmap("cividis")
    norm = Normalize(vmin, vmax)
    edges = np.linspace(vmin, vmax, N_BINS + 1)
    groups = {}
    for p, v in values.items():
        if np.isnan(v):
            continue
        b = int(np.clip(np.searchsorted(edges, v, side="right") - 1, 0, N_BINS - 1))
        groups.setdefault(b, []).append(p)
    colors = {b: [float(c) for c in list(cmap(norm((b + 0.5) / N_BINS)))[:3]]
              for b in range(N_BINS)}
    return groups, colors


def _scene(cmd):
    cmd.hide("everything")
    cmd.show("cartoon", "tem1")
    cmd.set("cartoon_transparency", 0)
    cmd.set("spec_reflect", 0)
    cmd.set("ambient", 0.15)
    cmd.set("direct", 0.6)
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("orthoscopic", 1)
    cmd.set("antialias", 2)
    cmd.hide("cartoon", "resi 1-%d" % NTERM)


def render(cmd, values, vmin, vmax, outfile, view=None, probe=False):
    from pymol import cmd as _c
    _c.reinitialize()
    _c.load(PDB, "tem1")
    _scene(_c)
    if view is None:
        _c.orient("tem1 and not resi 1-%d" % NTERM)
        _c.turn("x", -22)
        _c.turn("y", 28)
        _c.zoom("tem1 and not resi 1-%d" % NTERM, 2)
    else:
        _c.set_view(view)

    if probe:
        _c.hide("everything")
        _c.set_color("probe_s70", [1.0, 0.0, 0.0])
        _c.set_color("probe_omega", [0.0, 1.0, 0.0])
        _c.show("spheres", "resi %d and name CB" % CATALYTIC[0])
        _c.set("sphere_scale", 1.4)
        _c.color("probe_s70", "resi %d and name CB" % CATALYTIC[0])
        _c.show("spheres", "(resi %d-%d) and name CA" % (OMEGA[0], OMEGA[-1]))
        _c.set("sphere_scale", 0.9)
        _c.color("probe_omega", "(resi %d-%d) and name CA" % (OMEGA[0], OMEGA[-1]))
        _c.png(outfile, width=SIZE, height=SIZE, dpi=200, ray=1)
        return None

    groups, colors = _bins(values, vmin, vmax)
    _c.set_color("nores", [0.74, 0.74, 0.74])
    for b, ps in groups.items():
        _c.set_color("bin_%d" % b, colors[b])
        _c.color("bin_%d" % b, "resi " + "+".join(str(p) for p in ps))
    _c.set_color("omega_hi", list(OMEGA_RGB))
    _c.color("omega_hi", "resi %d-%d" % (OMEGA[0], OMEGA[-1]))
    _c.set_color("s70", list(S70_RGB))
    _c.show("spheres", "resi %d and name CB" % CATALYTIC[0])
    _c.set("sphere_scale", 1.2)
    _c.color("s70", "resi %d and name CB" % CATALYTIC[0])
    _c.png(outfile, width=SIZE, height=SIZE, dpi=200, ray=1)
    return _c.get_view()


def bbox_of_molecule(path, margin=90):
    """找非白像素的外接框（用于裁掉大片留白）。"""
    from PIL import Image
    im = np.array(Image.open(path).convert("RGB")).astype(int)
    nonwhite = (im.sum(axis=2) < 720)
    ys, xs = np.nonzero(nonwhite)
    H, W = im.shape[:2]
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(W, int(xs.max()) + margin + 1)
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(H, int(ys.max()) + margin + 1)
    return im, (x0, y0, x1, y1)


if __name__ == "__main__":
    from pymol import cmd
    import matplotlib
    matplotlib.use("Agg")

    d_vals = np.array([v for v in del_val.values() if not np.isnan(v)])
    b_vals = np.array(list(bur_val.values()))
    dmin, dmax = float(d_vals.min()), float(d_vals.max())
    bmin, bmax = float(b_vals.min()), float(b_vals.max())

    fA = os.path.join(OUTDIR, "tem1_deleteriousness.png")
    fB = os.path.join(OUTDIR, "tem1_burial.png")
    fP = os.path.join(OUTDIR, "tem1_probe.png")
    view = render(cmd, del_val, dmin, dmax, fA)
    render(cmd, bur_val, bmin, bmax, fB, view=view)
    render(cmd, del_val, dmin, dmax, fP, view=view, probe=True)

    from PIL import Image
    imA, box = bbox_of_molecule(fA)
    x0, y0, x1, y1 = box
    for src, dst in ((fA, os.path.join(OUTDIR, "crop_deleteriousness.png")),
                     (fB, os.path.join(OUTDIR, "crop_burial.png")),
                     (fP, os.path.join(OUTDIR, "crop_probe.png"))):
        Image.open(src).crop(box).save(dst)

    # 探针图 → Ser70 / Ω-loop 的像素位置（按色相判别，不要求纯色）
    probe = np.array(Image.open(os.path.join(OUTDIR, "crop_probe.png")).convert("RGB")).astype(int)
    R, G, B = probe[:, :, 0], probe[:, :, 1], probe[:, :, 2]

    def centroid(mask):
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        return float(xs.mean()), float(ys.mean()), int(len(xs))

    m_red = (R > 120) & (R > G + 60) & (R > B + 60)
    m_grn = (G > 120) & (G > R + 60) & (G > B + 60)
    c70 = centroid(m_red)
    com = centroid(m_grn)
    np.savez(os.path.join(OUTDIR, "probe_positions.npz"),
             s70_px=np.array(c70[:2]) if c70 else np.array([np.nan, np.nan]),
             omega_px=np.array(com[:2]) if com else np.array([np.nan, np.nan]),
             crop=np.array(box), width=x1 - x0, height=y1 - y0,
             dmin=dmin, dmax=dmax, bmin=bmin, bmax=bmax)
    print("deleteriousness range: %.3f .. %.3f" % (dmin, dmax))
    print("burial range        : %.1f .. %.1f" % (bmin, bmax))
    print("crop box:", box, "-> %dx%d" % (x1 - x0, y1 - y0))
    print("Ser70 px:", c70, " Omega px:", com)
