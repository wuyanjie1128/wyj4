# app.py
# Web UI for the interactive poster generator (Streamlit version)

import io
import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import streamlit as st

# -------------------------
# Page / Layout
# -------------------------
st.set_page_config(page_title="Interactive Poster • Web", page_icon="🎨", layout="wide")

st.title("🎨 Interactive Poster • Web")
st.caption("支持 CSV 调色板，参数可调，实时预览，并可下载 PNG")

# -------------------------
# Core: blob (fallback from your notebook)
# -------------------------
def blob(center=(0.5, 0.5), r=0.3, points=240, wobble=0.15, phase_shift=0.0):
    angles = np.linspace(0, 2*math.pi, points, endpoint=False)
    angles = angles + phase_shift * np.sin(3*angles)
    radii = r * (1 + wobble*(np.random.rand(points)-0.5))
    x = center[0] + radii*np.cos(angles)
    y = center[1] + radii*np.sin(angles)
    return x, y

PALETTE_CSV = "palette.csv"

# -------------------------
# Palette helpers
# -------------------------
def _normalize_rgb_tuple(rgb):
    r, g, b = rgb
    # 支持 0..255 或 0..1
    if max(r, g, b) > 1:
        r, g, b = r/255.0, g/255.0, b/255.0
    return (float(r), float(g), float(b))

def load_palette_from_csv_like(file_or_path):
    """
    接受路径或已上传的文件对象，读取列 R/G/B（大小写不敏感）。
    返回 [(r,g,b), ...]，若失败则返回空列表。
    """
    try:
        if file_or_path is None:
            return []
        if hasattr(file_or_path, "read"):
            df = pd.read_csv(file_or_path)
        else:
            if not os.path.exists(file_or_path):
                return []
            df = pd.read_csv(file_or_path)
        cols = {c.lower(): c for c in df.columns}
        if all(k in cols for k in ("r", "g", "b")) and len(df) > 0:
            vals = df[[cols["r"], cols["g"], cols["b"]]].values.tolist()
            return [_normalize_rgb_tuple(t) for t in vals]
    except Exception:
        pass
    return []

def pastel_fallback():
    return [(0.75,0.85,0.98),(0.81,0.90,0.84),(0.98,0.92,0.76),(0.86,0.80,0.90)]

def _gen_pastel(n):
    return [
        (random.uniform(0.4,0.9), random.uniform(0.4,0.9), random.uniform(0.4,0.9))
        for _ in range(n)
    ]

def _gen_vivid(n):
    cols = []
    for _ in range(n):
        hi = random.randint(0,2)
        c = [random.uniform(0.0,0.5) for _ in range(3)]
        c[hi] = random.uniform(0.7,1.0)
        cols.append(tuple(c))
    return cols

def _gen_mono(n, base=None):
    if base is None:
        base = (random.random(), random.random(), random.random())
    br, bg, bb = base
    cols = []
    for i in range(n):
        f = 0.4 + 0.6*(i/max(1, n-1))
        cols.append((br*f, bg*f, bb*f))
    return cols

def _gen_random(n):
    return [(random.random(), random.random(), random.random()) for _ in range(n)]

def get_palette_by_mode(mode: str, n_colors: int, seed: int, csv_uploaded=None):
    """
    与笔记本逻辑一致：mode 为 csv 时优先使用上传文件，
    否则尝试本地 palette.csv，失败则给出柔和的默认值。
    """
    random.seed(seed); np.random.seed(seed)
    m = (mode or "").lower()
    if m == "csv":
        pal = []
        # 1) 上传优先
        if csv_uploaded is not None:
            pal = load_palette_from_csv_like(csv_uploaded)
        # 2) 退回本地 palette.csv
        if not pal:
            pal = load_palette_from_csv_like(PALETTE_CSV)
        # 3) 再退回默认柔和色
        if not pal:
            pal = pastel_fallback()
        return pal
    if m == "pastel":  return _gen_pastel(n_colors)
    if m == "vivid":   return _gen_vivid(n_colors)
    if m == "mono":    return _gen_mono(n_colors)
    if m == "random":  return _gen_random(n_colors)
    return _gen_pastel(n_colors)

# -------------------------
# Drawing
# -------------------------
def draw_poster(n_layers=8, wobble=0.15, palette_mode="pastel", seed=0, csv_uploaded=None, figsize=(7,7)):
    random.seed(seed); np.random.seed(seed)
    palette = get_palette_by_mode(palette_mode, n_layers, seed, csv_uploaded)

    fig, ax = plt.subplots(figsize=figsize)
    patches, colors = [], []

    for i in range(int(n_layers)):
        cx, cy = random.uniform(0.15,0.85), random.uniform(0.15,0.85)
        r = random.uniform(0.18, 0.32)
        phase = random.uniform(0.0, 1.0)
        x, y = blob(center=(cx,cy), r=r, points=260, wobble=wobble*(1+0.15*i), phase_shift=phase)
        patches.append(Polygon(np.column_stack((x, y)), closed=True))
        colors.append(palette[i % len(palette)])

    coll = PatchCollection(patches, alpha=0.68)
    coll.set_facecolor(colors); coll.set_edgecolor("none")
    ax.add_collection(coll)

    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"Interactive Poster • {palette_mode}", loc="left", fontsize=16, fontweight="bold")
    return fig, palette

# -------------------------
# Sidebar Controls
# -------------------------
with st.sidebar:
    st.header("⚙️ 参数设置")
    n_layers = st.slider("Layers", min_value=3, max_value=20, value=8, step=1)
    wobble = st.slider("Wobble", min_value=0.01, max_value=0.5, value=0.15, step=0.01)
    palette_mode = st.selectbox("palette_mode", options=["pastel","vivid","mono","random","csv"], index=0)
    seed = st.number_input("Seed", min_value=0, max_value=999999, value=0, step=1)

    csv_uploaded = None
    if palette_mode == "csv":
        st.markdown("**CSV 要求**：包含列 `R,G,B`（0..1 或 0..255 都可）。")
        csv_uploaded = st.file_uploader("上传 CSV（可选，若不上传则尝试本地 palette.csv）", type=["csv"])

    st.divider()
    export_dpi = st.slider("导出 PNG DPI", min_value=72, max_value=600, value=300, step=12)
    fig_size = st.select_slider("画布尺寸（英寸）", options=[(6,6),(7,7),(8,8),(10,10),(12,12)], value=(7,7))
    st.caption("尺寸仅影响导出及绘图布局，不改变视觉比例。")

# -------------------------
# Generate & Show
# -------------------------
fig, palette = draw_poster(
    n_layers=n_layers,
    wobble=wobble,
    palette_mode=palette_mode,
    seed=seed,
    csv_uploaded=csv_uploaded,
    figsize=fig_size
)

col_left, col_right = st.columns([3, 1], gap="large")

with col_left:
    st.pyplot(fig, clear_figure=True)

    # Download as PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=export_dpi, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    buf.seek(0)
    st.download_button("⬇️ 下载 PNG", data=buf, file_name=f"poster_{palette_mode}_seed{seed}.png", mime="image/png")

with col_right:
    st.subheader("当前调色板预览")
    # 将 palette 展示为色块
    if palette:
        # 显示前 max_n 个色块
        max_n = min(20, len(palette))
        for i in range(max_n):
            r, g, b = palette[i]
            hex_color = "#{:02X}{:02X}{:02X}".format(int(r*255), int(g*255), int(b*255))
            st.write(
                f"#{i+1} {hex_color}",
            )
            st.color_picker(label=f"颜色 {i+1}", value=hex_color, key=f"cp_{i}", disabled=True)
    else:
        st.info("未能加载到任何颜色，已使用默认柔和色。")

st.divider()
with st.expander("📄 CSV 说明 / 使用提示", expanded=False):
    st.markdown(
        """
**CSV 格式**  
- 需要包含列：`R,G,B`（大小写不敏感）  
- 取值范围可以是 `0..1` 或 `0..255`，会自动归一化。  
- 选择 `palette_mode=csv` 后：  
  1) 若你上传了 CSV，则优先使用；  
  2) 否则尝试读取本地 `palette.csv`；  
  3) 若仍失败，自动回退到默认柔和色。  

**常见问题**  
- 如果导出 PNG 过大或过小，请调整侧边栏的 “DPI” 或 “画布尺寸”。  
- 图像是随机生成的；固定 `Seed` 可复现相同图案。  
        """
    )
