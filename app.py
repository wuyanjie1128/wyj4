# -*- coding: utf-8 -*-
# Web UI for "Interactive Poster" with CSV palette support
# Run: streamlit run app.py

import io
import math
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import streamlit as st

PALETTE_CSV = "palette.csv"

# ----------------------------
# Utilities
# ----------------------------
def blob(center=(0.5, 0.5), r=0.3, points=240, wobble=0.15, phase_shift=0.0):
    angles = np.linspace(0, 2 * math.pi, points, endpoint=False)
    angles = angles + phase_shift * np.sin(3 * angles)
    radii = r * (1 + wobble * (np.random.rand(points) - 0.5))
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    return x, y


def _gen_pastel(n):
    return [
        (
            random.uniform(0.4, 0.9),
            random.uniform(0.4, 0.9),
            random.uniform(0.4, 0.9),
        )
        for _ in range(n)
    ]


def _gen_vivid(n):
    cols = []
    for _ in range(n):
        hi = random.randint(0, 2)
        c = [random.uniform(0.0, 0.5) for _ in range(3)]
        c[hi] = random.uniform(0.7, 1.0)
        cols.append(tuple(c))
    return cols


def _gen_mono(n, base=None):
    if base is None:
        base = (random.random(), random.random(), random.random())
    br, bg, bb = base
    cols = []
    for i in range(n):
        f = 0.4 + 0.6 * (i / max(1, n - 1))
        cols.append((br * f, bg * f, bb * f))
    return cols


def _gen_random(n):
    return [(random.random(), random.random(), random.random()) for _ in range(n)]


def parse_palette_df(df: pd.DataFrame):
    """Accept columns named R/G/B (case-insensitive); values in 0..1 or 0..255."""
    cols = {c.lower(): c for c in df.columns}
    if not all(k in cols for k in ("r", "g", "b")):
        return []
    vals = df[[cols["r"], cols["g"], cols["b"]]].values.tolist()
    out = []
    for r, g, b in vals:
        if max(r, g, b) > 1:
            r, g, b = r / 255.0, g / 255.0, b / 255.0
        out.append((float(r), float(g), float(b)))
    return out


def load_palette_from_csv_path(path: str):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            parsed = parse_palette_df(df)
            if parsed:
                return parsed
        except Exception:
            pass
    # fallback pastel
    return [(0.75, 0.85, 0.98), (0.81, 0.90, 0.84), (0.98, 0.92, 0.76), (0.86, 0.80, 0.90)]


def get_palette_by_mode(mode: str, n_colors: int, seed: int, uploaded_df: pd.DataFrame | None):
    random.seed(seed)
    np.random.seed(seed)
    m = (mode or "").lower()

    if m == "csv":
        # Priority: uploaded CSV -> local palette.csv -> fallback
        if uploaded_df is not None:
            parsed = parse_palette_df(uploaded_df)
            if parsed:
                return parsed
        return load_palette_from_csv_path(PALETTE_CSV)

    if m == "pastel":
        return _gen_pastel(n_colors)
    if m == "vivid":
        return _gen_vivid(n_colors)
    if m == "mono":
        return _gen_mono(n_colors)
    if m == "random":
        return _gen_random(n_colors)
    return _gen_pastel(n_colors)


def draw_poster(n_layers=8, wobble=0.15, palette_mode="pastel", seed=0, uploaded_df: pd.DataFrame | None = None):
    random.seed(seed)
    np.random.seed(seed)
    palette = get_palette_by_mode(palette_mode, n_layers, seed, uploaded_df)

    fig, ax = plt.subplots(figsize=(7, 7))
    patches, colors = [], []

    for i in range(int(n_layers)):
        cx, cy = random.uniform(0.15, 0.85), random.uniform(0.15, 0.85)
        r = random.uniform(0.18, 0.32)
        phase = random.uniform(0.0, 1.0)
        x, y = blob(center=(cx, cy), r=r, points=260, wobble=wobble * (1 + 0.15 * i), phase_shift=phase)
        patches.append(Polygon(np.column_stack((x, y)), closed=True))
        colors.append(palette[i % len(palette)])

    coll = PatchCollection(patches, alpha=0.68)
    coll.set_facecolor(colors)
    coll.set_edgecolor("none")
    ax.add_collection(coll)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Interactive Poster • {palette_mode}", loc="left", fontsize=16, fontweight="bold")

    return fig


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Interactive Poster", layout="wide")
st.title("🎨 Interactive Poster (Web)")
st.caption("支持从 CSV 读取调色板（R,G,B 列；值可为 0..1 或 0..255）。")

with st.sidebar:
    st.header("Controls")
    n_layers = st.slider("Layers", min_value=3, max_value=20, value=8, step=1)
    wobble = st.slider("Wobble", min_value=0.01, max_value=0.50, value=0.15, step=0.01)
    palette_mode = st.selectbox("palette_mode", options=["pastel", "vivid", "mono", "random", "csv"], index=0)
    seed = st.number_input("Seed", min_value=0, max_value=999999, value=0, step=1)

    uploaded_df = None
    uploaded_file = None
    if palette_mode == "csv":
        uploaded_file = st.file_uploader("上传 palette.csv（含 R,G,B 列）", type=["csv"])
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                st.success("已载入上传的 CSV 调色板。")
            except Exception as e:
                st.error(f"CSV 解析失败：{e}")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔀 随机种子"):
            seed = random.randint(0, 999999)
            st.session_state["_seed_override"] = seed
    with col_btn2:
        if st.button("♻️ 重绘"):
            pass  # Streamlit 会根据控件状态自动重绘

# 若随机按钮更新了 seed，则覆盖展示
if "_seed_override" in st.session_state:
    seed = int(st.session_state["_seed_override"])
    st.info(f"当前种子：{seed}")

# 绘制
fig = draw_poster(
    n_layers=n_layers,
    wobble=wobble,
    palette_mode=palette_mode,
    seed=seed,
    uploaded_df=uploaded_df,
)

st.pyplot(fig, use_container_width=True)

# 下载按钮
png_buf = io.BytesIO()
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
filename = f"poster_{palette_mode}_seed{seed}_{timestamp}.png"
fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)
st.download_button(
    label="⬇️ 下载 PNG",
    data=png_buf.getvalue(),
    file_name=filename,
    mime="image/png",
)

# CSV 提示信息
if palette_mode == "csv" and uploaded_df is None and not os.path.exists(PALETTE_CSV):
    st.warning("未检测到上传的 CSV，也未在工作目录发现 palette.csv。已使用内置柔和色作为后备方案。")
elif palette_mode == "csv" and uploaded_df is None and os.path.exists(PALETTE_CSV):
    st.success("已从本地文件 palette.csv 载入调色板。")
