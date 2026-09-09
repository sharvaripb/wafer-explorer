from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

st.set_page_config(page_title="Wafer Explorer", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #eeeeee; }
label { color: black !important; font-weight: 600 !important; }
div[data-testid="stToggle"] * { color: black !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color:black;'>Wafer Explorer</h1>", unsafe_allow_html=True)
st.caption("Portfolio reproduction using synthetic wafer-map data and synthetic image embeddings.")

show_cluster_outlines = st.session_state.get("show_cluster_outlines", True)
col1, col2 = st.columns([3, 1])

with col1:
    feature_file = st.selectbox(
        "Feature Set",
        ["dinov3s_feat_284_alg.csv", "dinov3b_feat_284_alg.csv"],
    )
    clustering_method = st.selectbox(
        "Clustering Method",
        ["K-Means", "Agglomerative"],
    )
    n_clusters = st.number_input(
        "Number of Clusters",
        min_value=2,
        max_value=10,
        value=3,
        step=1,
    )

def parse_tensor(value):
    value = value.replace("tensor([", "").replace("])", "")
    return np.fromstring(value, sep=",")

@st.cache_data
def load_data(feature_file, clustering_method, n_clusters):
    df = pd.read_csv(DATA_DIR / feature_file)
    X = np.vstack(df["FeatDv3Cls"].apply(parse_tensor))
    X_scaled = StandardScaler().fit_transform(X)

    coords = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30,
        init="pca",
        learning_rate="auto",
    ).fit_transform(X_scaled)

    if clustering_method == "K-Means":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")

    df["cluster"] = model.fit_predict(X_scaled)
    df["tsne_x"] = coords[:, 0]
    df["tsne_y"] = coords[:, 1]
    return df

@st.cache_data
def make_wafer_png(pattern, seed, size=92):
    rng = np.random.default_rng(int(seed))
    grid = 25
    margin = 6
    cell = (size - 2 * margin) / grid

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx = cy = (grid - 1) / 2
    radius = grid * 0.47

    defect = set()
    valid = []

    for row in range(grid):
        for col in range(grid):
            if (col - cx) ** 2 + (row - cy) ** 2 <= radius ** 2:
                valid.append((row, col))

    if pattern == "center":
        defect = {
            p for p in valid
            if (p[1] - cx) ** 2 + (p[0] - cy) ** 2 < (radius * 0.30) ** 2
            and rng.random() < 0.72
        }
    elif pattern == "ring":
        defect = {
            p for p in valid
            if radius * 0.62 < np.hypot(p[1] - cx, p[0] - cy) < radius * 0.86
            and rng.random() < 0.58
        }
    elif pattern == "edge":
        angle0 = rng.uniform(-np.pi, np.pi)
        defect = {
            p for p in valid
            if np.hypot(p[1] - cx, p[0] - cy) > radius * 0.70
            and abs(np.angle(np.exp(1j * (np.arctan2(p[0]-cy, p[1]-cx) - angle0)))) < 0.65
            and rng.random() < 0.75
        }
    elif pattern == "scratch":
        slope = rng.uniform(-0.8, 0.8)
        offset = rng.uniform(-3, 3)
        defect = {
            p for p in valid
            if abs((p[0] - cy) - slope * (p[1] - cx) - offset) < 1.1
            and rng.random() < 0.82
        }
    elif pattern == "local":
        gx, gy = rng.uniform(-5, 5, size=2)
        defect = {
            p for p in valid
            if (p[1] - cx - gx) ** 2 + (p[0] - cy - gy) ** 2 < rng.uniform(7, 16)
            and rng.random() < 0.82
        }
    elif pattern == "random":
        defect = {p for p in valid if rng.random() < 0.12}
    else:
        defect = {p for p in valid if rng.random() < 0.018}

    for row, col in valid:
        x0 = margin + col * cell
        y0 = margin + row * cell
        x1 = x0 + cell - 0.65
        y1 = y0 + cell - 0.65
        color = (38, 40, 43, 255) if (row, col) in defect else (214, 216, 219, 255)
        draw.rectangle((x0, y0, x1, y1), fill=color)

    # Small bottom notch.
    draw.rectangle(
        (
            size / 2 - 4,
            size - margin - cell * 1.1,
            size / 2 + 4,
            size - margin + 1,
        ),
        fill=(0, 0, 0, 0),
    )

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def wafer_image(pattern, seed):
    return Image.open(BytesIO(make_wafer_png(pattern, seed))).copy()

@st.cache_resource
def build_tsne_figure(feature_file, clustering_method, n_clusters, show_cluster_outlines):
    df = load_data(feature_file, clustering_method, n_clusters)

    cluster_colors = [
        "#008080", "#800020", "#007BA7", "#6A0DAD", "#D97706",
        "#2E8B57", "#C71585", "#64748B", "#B8860B", "#111827",
    ]
    colors = [cluster_colors[int(c) % len(cluster_colors)] for c in df["cluster"]]

    x_range = max(df["tsne_x"].max() - df["tsne_x"].min(), 1)
    y_range = max(df["tsne_y"].max() - df["tsne_y"].min(), 1)
    image_w = x_range * 0.023
    image_h = y_range * 0.034

    fig = go.Figure()

    if show_cluster_outlines:
        fig.add_trace(go.Scatter(
            x=df["tsne_x"],
            y=df["tsne_y"],
            mode="markers",
            marker=dict(
                symbol="square",
                size=20,
                color=colors,
                opacity=0.95,
                line=dict(width=1.2, color="white"),
            ),
            text=df["labels"],
            customdata=df["cluster"],
            hovertemplate="<b>%{text}</b><br>Cluster: %{customdata}<extra></extra>",
            showlegend=False,
        ))

    fig.add_trace(go.Scatter(
        x=df["tsne_x"],
        y=df["tsne_y"],
        mode="markers",
        marker=dict(size=20, opacity=0),
        text=df["labels"],
        customdata=df["cluster"],
        hovertemplate="<b>%{text}</b><br>Cluster: %{customdata}<extra></extra>",
        showlegend=False,
    ))

    for _, row in df.iterrows():
        fig.add_layout_image(dict(
            source=wafer_image(row["pattern"], row["seed"]),
            x=row["tsne_x"],
            y=row["tsne_y"],
            sizex=image_w,
            sizey=image_h,
            opacity=1,
            xref="x",
            yref="y",
            layer="above",
            xanchor="center",
            yanchor="middle",
        ))

    if show_cluster_outlines:
        for cluster_id in sorted(df["cluster"].unique()):
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    symbol="square",
                    size=12,
                    color=cluster_colors[int(cluster_id) % len(cluster_colors)],
                ),
                name=f"Cluster {cluster_id}",
            ))

    fig.update_layout(
        height=760,
        plot_bgcolor="#eeeeee",
        paper_bgcolor="#eeeeee",
        font=dict(color="black"),
        margin=dict(l=20, r=10, t=40, b=20),
        title=dict(
            text=f"{feature_file} | {clustering_method} clustering",
            font=dict(color="black", size=16),
        ),
        legend=dict(
            title=dict(text="Cluster assignment", font=dict(color="black")),
            font=dict(color="black"),
            bgcolor="rgba(238,238,238,0.82)",
        ),
        dragmode="select",
    )
    fig.update_xaxes(showgrid=False, zeroline=False, title="t-SNE 1")
    fig.update_yaxes(showgrid=False, zeroline=False, title="t-SNE 2")
    return fig

with st.spinner("Computing t-SNE and clustering..."):
    df = load_data(feature_file, clustering_method, n_clusters)

with col1:
    fig = build_tsne_figure(
        feature_file,
        clustering_method,
        n_clusters,
        show_cluster_outlines,
    )
    event = st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"tsne_{feature_file}_{clustering_method}_{n_clusters}_{show_cluster_outlines}",
        on_select="rerun",
        selection_mode="points",
    )

with col2:
    st.markdown("<h3 style='color:black;'>Wafer Preview</h3>", unsafe_allow_html=True)

    selected_idx = 0
    if event.selection.points:
        idx = event.selection.points[0].get("point_index", 0)
        if 0 <= idx < len(df):
            selected_idx = idx

    selected = df.iloc[selected_idx]
    st.image(
        wafer_image(selected["pattern"], selected["seed"]),
        use_container_width=True,
    )

    st.markdown(
        f"""
        <div style="color:black;font-size:18px;font-weight:600;text-align:center;margin-top:8px;">
        {selected["labels"]}<br>Cluster: {selected["cluster"]}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<h3 style='color:black;'>Cluster Counts</h3>", unsafe_allow_html=True)
    st.dataframe(
        df["cluster"]
        .value_counts()
        .sort_index()
        .rename_axis("Cluster")
        .reset_index(name="Count"),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "<div style='color:black;font-weight:600;margin-top:18px;'>Display Options</div>",
        unsafe_allow_html=True,
    )
    st.toggle(
        "Show cluster outlines",
        value=True,
        key="show_cluster_outlines",
    )
