from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering

st.set_page_config(page_title="Wafer Explorer", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
IMAGE_FOLDER = BASE_DIR / "wafer_images"
DATA_FOLDER = BASE_DIR / "data"

st.markdown("""
<style>
.stApp {
    background-color: #ececec;
}

label {
    color: black !important;
    font-weight: 600 !important;
}

div[data-testid="stToggle"] * {
    color: black !important;


</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color:black;'>Wafer Explorer</h1>", unsafe_allow_html=True)
st.caption(
    "Portfolio reproduction using synthetic wafer maps and synthetic feature embeddings. "
    "No proprietary manufacturing data is included."
)

show_cluster_outlines = st.session_state.get("show_cluster_outlines", True)

col1, col2 = st.columns([3, 1])

with col1:
    feature_file = st.selectbox(
        "Feature Set",
        ["dinov3s_feat_284_alg.csv", "dinov3b_feat_284_alg.csv"]
    )

    clustering_method = st.selectbox(
        "Clustering Method",
        ["K-Means", "Agglomerative"]
    )

    n_clusters = st.number_input(
        "Number of Clusters",
        min_value=2,
        max_value=10,
        value=3,
        step=1
    )

def parse_tensor(x):
    x = x.replace("tensor([", "")
    x = x.replace("])", "")
    return np.fromstring(x, sep=",")

@st.cache_data
def load_data(feature_file, clustering_method, n_clusters):
    df = pd.read_csv(DATA_FOLDER / feature_file)

    X = np.vstack(df["FeatDv3Cls"].apply(parse_tensor))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30
    )

    X_tsne = tsne.fit_transform(X_scaled)

    if clustering_method == "K-Means":
        cluster_model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
    else:
        cluster_model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward"
        )

    cluster_labels = cluster_model.fit_predict(X_scaled)

    df["tsne_x"] = X_tsne[:, 0]
    df["tsne_y"] = X_tsne[:, 1]
    df["cluster"] = cluster_labels

    df["image_path"] = df["labels"].apply(
        lambda x: str(IMAGE_FOLDER / f"{x}.bmp")
    )

    return df

@st.cache_data
def load_thumbnail(image_path):
    img = Image.open(image_path)
    img.thumbnail((128, 128))
    return img.copy()

@st.cache_resource
def build_tsne_figure(feature_file, clustering_method, n_clusters, show_cluster_outlines):
    df = load_data(feature_file, clustering_method, n_clusters)

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
        "#111827"
    ]

    colors = [
        cluster_colors[int(c) % len(cluster_colors)]
        for c in df["cluster"]
    ]

    fig = go.Figure()

    if show_cluster_outlines:
        fig.add_trace(
            go.Scatter(
                x=df["tsne_x"],
                y=df["tsne_y"],
                mode="markers",
                marker=dict(
                    symbol="square",
                    size=40,
                    color=colors,
                    opacity=0.95,
                    line=dict(
                        width=1.5,
                        color="white"
                    )
                ),
                text=df["labels"],
                customdata=df["cluster"],
                hovertemplate="<b>%{text}</b><br>Cluster: %{customdata}<extra></extra>",
                showlegend=False
            )
        )

    fig.add_trace(
        go.Scatter(
            x=df["tsne_x"],
            y=df["tsne_y"],
            mode="markers",
            marker=dict(
                size=45,
                opacity=0
            ),
            text=df["labels"],
            customdata=df["cluster"],
            hovertemplate="<b>%{text}</b><br>Cluster: %{customdata}<extra></extra>",
            showlegend=False
        )
    )

    for _, row in df.iterrows():
        img = load_thumbnail(row["image_path"])

        fig.add_layout_image(
            dict(
                source=img,
                x=row["tsne_x"],
                y=row["tsne_y"],
                sizex=2,
                sizey=2,
                opacity=1,
                xref="x",
                yref="y",
                layer="above",
                xanchor="center",
                yanchor="middle"
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
                        size=14,
                        color=cluster_colors[int(cluster_id) % len(cluster_colors)]
                    ),
                    name=f"Cluster {cluster_id}"
                )
            )

    fig.update_layout(
        height=800,
        plot_bgcolor="#ececec",
        paper_bgcolor="#ececec",
        font=dict(color="black"),
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(
            text=f"{feature_file} | {clustering_method} clustering",
            font=dict(
                color="black",
                size=18
            )
        ),
        legend=dict(
            title=dict(
                text="Cluster assignment",
                font=dict(color="black")
            ),
            font=dict(
                color="black"
            ),
            bgcolor="rgba(236,236,236,0.8)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1
        )
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        title="t-SNE 1"
    )

    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        title="t-SNE 2"
    )

    return fig

with st.spinner("Computing t-SNE and clustering from selected feature file..."):
    df = load_data(feature_file, clustering_method, n_clusters)

with col1:
    fig = build_tsne_figure(
        feature_file,
        clustering_method,
        n_clusters,
        show_cluster_outlines
    )

    event = st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"tsne_plot_{feature_file}_{clustering_method}_{n_clusters}_{show_cluster_outlines}",
        on_select="rerun",
        selection_mode="points"
    )

with col2:
    st.markdown(
        "<h3 style='color:black;'>Wafer Preview</h3>",
        unsafe_allow_html=True
    )

    selected_label = df["labels"].iloc[0]

    if event.selection.points:
        point_index = event.selection.points[0]["point_index"]
        selected_label = df.iloc[point_index]["labels"]

    selected_row = df[df["labels"] == selected_label].iloc[0]

    st.image(
        selected_row["image_path"],
        use_container_width=True
    )

    st.markdown(
        f"""
        <div style="color:black; font-size:18px; font-weight:600;
        text-align:center; margin-top:10px;">
        {selected_label}<br>
        Cluster: {selected_row["cluster"]}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<h3 style='color:black;'>Cluster Counts</h3>",
        unsafe_allow_html=True
    )

    st.dataframe(
        df["cluster"]
        .value_counts()
        .sort_index()
        .rename_axis("Cluster")
        .reset_index(name="Count"),
        use_container_width=True
    )

    st.markdown(
        "<div style='color:black;font-weight:600;margin-top:18px;'>Display Options</div>",
        unsafe_allow_html=True
    )

    st.toggle(
        "Show cluster outlines",
        value=True,
        key="show_cluster_outlines"
    )