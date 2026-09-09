from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="Wafer Explorer", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #eeeeee; }
label { color: black !important; font-weight: 600 !important; }
div[data-testid="stToggle"] * { color: black !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color:black;'>Wafer Explorer</h1>", unsafe_allow_html=True)
st.caption(
    "Portfolio reproduction using synthetic wafer maps and synthetic image embeddings. "
    "No proprietary manufacturing data is included."
)

show_cluster_outlines = st.session_state.get("show_cluster_outlines", True)
col1, col2 = st.columns([3, 1])

with col1:
    feature_set = st.selectbox("Feature Set", ["DINOv3-S", "DINOv3-B"])
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


@st.cache_data
def generate_demo_data(feature_set):
    rng = np.random.default_rng(42 if feature_set == "DINOv3-S" else 84)
    n = 284

    # Continuous latent manifold rather than five artificially isolated blobs.
    t = np.sort(rng.uniform(-3.0, 3.0, n))
    latent = np.column_stack(
        [
            2.2 * t + 0.45 * np.sin(2.2 * t),
            1.5 * np.sin(1.05 * t) + 0.25 * np.sin(2.7 * t),
            np.cos(0.8 * t),
            np.sin(1.7 * t),
            (t ** 2) / 4.0,
            rng.normal(0, 0.75, n),
            rng.normal(0, 0.55, n),
            rng.normal(0, 0.35, n),
        ]
    )

    projection = rng.normal(0, 0.9, size=(latent.shape[1], 64))
    noise = 0.48 if feature_set == "DINOv3-S" else 0.36
    X = latent @ projection + rng.normal(0, noise, size=(n, 64))

    labels = [f"wafer_{i:03d}" for i in range(1, n + 1)]

    pattern_types = np.array(
        ["edge", "scratch", "local", "ring", "center", "random", "clean"]
    )
    band = np.floor(
        (t - t.min()) / (t.max() - t.min() + 1e-9) * len(pattern_types)
    ).astype(int)
    band = np.clip(band, 0, len(pattern_types) - 1)

    return pd.DataFrame(
        {
            "labels": labels,
            "pattern": pattern_types[band],
            "seed": np.arange(1000, 1000 + n),
        }
    ), X


@st.cache_data
def run_analysis(feature_set, clustering_method, n_clusters):
    df, X = generate_demo_data(feature_set)
    X_scaled = StandardScaler().fit_transform(X)

    coords = TSNE(
        n_components=2,
        random_state=42,
        perplexity=34,
        init="pca",
        learning_rate="auto",
    ).fit_transform(X_scaled)

    if clustering_method == "K-Means":
        model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=20,
        )
    else:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward",
        )

    df["cluster"] = model.fit_predict(X_scaled)
    df["tsne_x"] = coords[:, 0]
    df["tsne_y"] = coords[:, 1]
    return df


@st.cache_data
def make_wafer_png(pattern, seed, size=86):
    rng = np.random.default_rng(int(seed))
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    grid = 25
    margin = 5
    cell = (size - 2 * margin) / grid
    cx = cy = (grid - 1) / 2
    radius = grid * 0.47

    valid = []
    for row in range(grid):
        for col in range(grid):
            if (col - cx) ** 2 + (row - cy) ** 2 <= radius ** 2:
                valid.append((row, col))

    if pattern == "center":
        defect = {
            p for p in valid
            if (p[1] - cx) ** 2 + (p[0] - cy) ** 2 < (radius * 0.28) ** 2
            and rng.random() < 0.72
        }
    elif pattern == "ring":
        defect = {
            p for p in valid
            if radius * 0.60 < np.hypot(p[1] - cx, p[0] - cy) < radius * 0.84
            and rng.random() < 0.54
        }
    elif pattern == "edge":
        angle0 = rng.uniform(-np.pi, np.pi)
        defect = {
            p for p in valid
            if np.hypot(p[1] - cx, p[0] - cy) > radius * 0.69
            and abs(
                np.angle(
                    np.exp(
                        1j
                        * (
                            np.arctan2(p[0] - cy, p[1] - cx)
                            - angle0
                        )
                    )
                )
            ) < 0.60
            and rng.random() < 0.72
        }
    elif pattern == "scratch":
        slope = rng.uniform(-0.8, 0.8)
        offset = rng.uniform(-3.0, 3.0)
        defect = {
            p for p in valid
            if abs((p[0] - cy) - slope * (p[1] - cx) - offset) < 1.0
            and rng.random() < 0.80
        }
    elif pattern == "local":
        gx, gy = rng.uniform(-5, 5, size=2)
        spread = rng.uniform(8, 15)
        defect = {
            p for p in valid
            if (p[1] - cx - gx) ** 2 + (p[0] - cy - gy) ** 2 < spread
            and rng.random() < 0.82
        }
    elif pattern == "random":
        defect = {p for p in valid if rng.random() < 0.11}
    else:
        defect = {p for p in valid if rng.random() < 0.015}

    for row, col in valid:
        x0 = margin + col * cell
        y0 = margin + row * cell
        x1 = x0 + cell - 0.55
        y1 = y0 + cell - 0.55
        fill = (
            (42, 42, 44, 255)
            if (row, col) in defect
            else (211, 213, 216, 255)
        )
        draw.rectangle((x0, y0, x1, y1), fill=fill)

    draw.rectangle(
        (
            size / 2 - 4,
            size - margin - cell * 1.2,
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
def build_tsne_figure(
    feature_set,
    clustering_method,
    n_clusters,
    show_cluster_outlines,
):
    df = run_analysis(feature_set, clustering_method, n_clusters)

    cluster_colors = [
        "#008080",
        "#800020",
        "#007BA7",
        "#6A0DAD",
        "#D97706",
        "#2E8B57",
        "#C71585",
        "#64748B",
        "#B8860B",
        "#111827",
    ]

    colors = [
        cluster_colors[int(cluster) % len(cluster_colors)]
        for cluster in df["cluster"]
    ]

    x_span = max(df["tsne_x"].max() - df["tsne_x"].min(), 1)
    y_span = max(df["tsne_y"].max() - df["tsne_y"].min(), 1)
    image_w = x_span * 0.018
    image_h = y_span * 0.026

    fig = go.Figure()

    if show_cluster_outlines:
        fig.add_trace(
            go.Scatter(
                x=df["tsne_x"],
                y=df["tsne_y"],
                mode="markers",
                marker=dict(
                    symbol="square",
                    size=16,
                    color=colors,
                    opacity=0.90,
                    line=dict(width=0.9, color="white"),
                ),
                text=df["labels"],
                customdata=df["cluster"],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Cluster: %{customdata}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=df["tsne_x"],
            y=df["tsne_y"],
            mode="markers",
            marker=dict(size=17, opacity=0),
            text=df["labels"],
            customdata=df["cluster"],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Cluster: %{customdata}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    for _, row in df.iterrows():
        fig.add_layout_image(
            dict(
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
            )
        )

    if show_cluster_outlines:
        for cluster_id in sorted(df["cluster"].unique()):
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(
                        symbol="square",
                        size=11,
                        color=cluster_colors[
                            int(cluster_id) % len(cluster_colors)
                        ],
                    ),
                    name=f"Cluster {cluster_id}",
                )
            )

    fig.update_layout(
        height=760,
        plot_bgcolor="#eeeeee",
        paper_bgcolor="#eeeeee",
        font=dict(color="black"),
        margin=dict(l=20, r=10, t=40, b=20),
        title=dict(
            text=f"{feature_set} | {clustering_method} clustering",
            font=dict(color="black", size=16),
        ),
        legend=dict(
            title=dict(
                text="Cluster assignment",
                font=dict(color="black"),
            ),
            font=dict(color="black"),
            bgcolor="rgba(238,238,238,0.82)",
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        title="t-SNE 1",
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        title="t-SNE 2",
    )

    return fig


with st.spinner("Computing t-SNE and clustering..."):
    df = run_analysis(feature_set, clustering_method, n_clusters)

with col1:
    fig = build_tsne_figure(
        feature_set,
        clustering_method,
        n_clusters,
        show_cluster_outlines,
    )

    event = st.plotly_chart(
        fig,
        use_container_width=True,
        key=(
            f"tsne_{feature_set}_{clustering_method}_"
            f"{n_clusters}_{show_cluster_outlines}"
        ),
        on_select="rerun",
        selection_mode="points",
    )

with col2:
    st.markdown(
        "<h3 style='color:black;'>Wafer Preview</h3>",
        unsafe_allow_html=True,
    )

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
        <div style="
            color:black;
            font-size:18px;
            font-weight:600;
            text-align:center;
            margin-top:8px;">
        {selected["labels"]}<br>
        Cluster: {selected["cluster"]}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h3 style='color:black;'>Cluster Counts</h3>",
        unsafe_allow_html=True,
    )

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
        "<div style='color:black;font-weight:600;margin-top:18px;'>"
        "Display Options</div>",
        unsafe_allow_html=True,
    )

    st.toggle(
        "Show cluster outlines",
        value=True,
        key="show_cluster_outlines",
    )
