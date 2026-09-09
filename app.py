from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Wafer Explorer",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Inter+Tight:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: "Inter", sans-serif;
    }

    .stApp {
        background: #F4F5F6;
        color: #181A1D;
    }

    /* Main page width */
    .block-container {
        max-width: 1500px;
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* --------------------------------------------------------
       HEADER PANEL
       -------------------------------------------------------- */

    .header-panel {
        width: 100%;
        box-sizing: border-box;
        background: #FFFFFF;
        border: 1px solid #D9DDE2;
        border-radius: 7px;
        padding: 24px 26px 21px 26px;
        margin: 0 0 18px 0;
        position: relative;
        overflow: hidden;
    }

    .header-panel::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: #00A6A6;
    }

    .wafer-title {
        font-family: "Inter Tight", "Helvetica Neue", Arial, sans-serif;
        font-size: 50px;
        line-height: 1.02;
        letter-spacing: -2.2px;
        font-weight: 500;
        color: #181A1D;
        margin: 0 0 8px 0;
    }

    .wafer-disclaimer {
        font-family: "Inter", sans-serif;
        font-size: 13px;
        line-height: 1.5;
        color: #747A82;
        margin: 0;
    }

    /* --------------------------------------------------------
       CONTROL LABELS
       -------------------------------------------------------- */

    label {
        font-family: "Inter", sans-serif !important;
        color: #181A1D !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        letter-spacing: 1.1px !important;
        text-transform: uppercase !important;
    }

    /* --------------------------------------------------------
       SELECT BOXES
       -------------------------------------------------------- */

    div[data-baseweb="select"] > div {
        background: #202328 !important;
        border: 1px solid #202328 !important;
        border-radius: 6px !important;
        min-height: 48px !important;
    }

    div[data-baseweb="select"] span {
        color: #F7F7F7 !important;
        font-family: "Inter", sans-serif !important;
    }

    div[data-baseweb="select"] svg {
        fill: #F7F7F7 !important;
    }

    /* --------------------------------------------------------
       NUMBER INPUT
       -------------------------------------------------------- */

    div[data-testid="stNumberInput"] input {
        background: #202328 !important;
        color: #F7F7F7 !important;
        border-color: #202328 !important;
        min-height: 48px !important;
        text-align: center;
        font-family: "Inter", sans-serif !important;
    }

    div[data-testid="stNumberInput"] button {
        background: #202328 !important;
        color: #F7F7F7 !important;
        border-color: #34383E !important;
        min-height: 48px !important;
    }

    div[data-testid="stNumberInput"] button:hover {
        background: #2A2E34 !important;
    }

    /* --------------------------------------------------------
       CARDS
       -------------------------------------------------------- */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF;
        border: 1px solid #D9DDE2 !important;
        border-radius: 7px !important;
        box-shadow: none !important;
    }

    /* --------------------------------------------------------
       SECTION HEADINGS
       -------------------------------------------------------- */

    .section-heading {
        font-family: "Inter", sans-serif;
        font-size: 13px;
        line-height: 1.2;
        font-weight: 700;
        letter-spacing: 1.35px;
        color: #181A1D;
        text-transform: uppercase;
        margin-top: 2px;
        margin-bottom: 8px;
    }

    .section-line {
        width: 44px;
        height: 4px;
        background: #00A6A6;
        border-radius: 2px;
        margin-bottom: 16px;
    }

    .selected-wafer {
        font-family: "Inter", sans-serif;
        color: #181A1D;
        font-size: 17px;
        font-weight: 600;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 1px;
    }

    .selected-cluster {
        font-family: "Inter", sans-serif;
        color: #666D75;
        font-size: 14px;
        font-weight: 500;
        text-align: center;
        margin-bottom: 14px;
    }

    /* --------------------------------------------------------
       TOGGLE
       -------------------------------------------------------- */

    div[data-testid="stToggle"] p {
        color: #181A1D !important;
        font-family: "Inter", sans-serif !important;
        font-size: 14px !important;
    }

    div[data-testid="stToggle"] button[aria-checked="true"] {
        background-color: #00A6A6 !important;
    }

    /* --------------------------------------------------------
       DATAFRAME
       -------------------------------------------------------- */

    div[data-testid="stDataFrame"] {
        border: 1px solid #D9DDE2;
        border-radius: 6px;
        overflow: hidden;
    }

    /* --------------------------------------------------------
       STREAMLIT CHROME
       -------------------------------------------------------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="header-panel">
        <div class="wafer-title">Wafer Explorer</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONTROLS
# ============================================================

with st.container(border=True):

    control_1, control_2, control_3 = st.columns(
        [1, 1, 1],
        gap="large",
    )

    with control_1:
        feature_set = st.selectbox(
            "Feature Set",
            ["DINOv3-S", "DINOv3-B"],
        )

    with control_2:
        clustering_method = st.selectbox(
            "Clustering Method",
            ["K-Means", "Agglomerative"],
        )

    with control_3:
        n_clusters = st.number_input(
            "Number of Clusters",
            min_value=2,
            max_value=10,
            value=3,
            step=1,
        )


st.write("")


# ============================================================
# SYNTHETIC EMBEDDING DATA
# ============================================================

@st.cache_data
def generate_demo_data(feature_set):

    rng = np.random.default_rng(
        42 if feature_set == "DINOv3-S" else 84
    )

    n = 284

    t = np.sort(
        rng.uniform(-3.0, 3.0, n)
    )

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

    projection = rng.normal(
        0,
        0.9,
        size=(latent.shape[1], 64),
    )

    noise = (
        0.48
        if feature_set == "DINOv3-S"
        else 0.36
    )

    X = (
        latent @ projection
        + rng.normal(
            0,
            noise,
            size=(n, 64),
        )
    )

    labels = [
        f"wafer_{i:03d}"
        for i in range(1, n + 1)
    ]

    pattern_types = np.array(
        [
            "edge",
            "texture",
            "local",
            "band",
            "center",
            "streak",
            "clean",
        ]
    )

    normalized_t = (
        (t - t.min())
        / (t.max() - t.min() + 1e-9)
    )

    pattern_idx = np.floor(
        normalized_t * len(pattern_types)
    ).astype(int)

    pattern_idx = np.clip(
        pattern_idx,
        0,
        len(pattern_types) - 1,
    )

    df = pd.DataFrame(
        {
            "labels": labels,
            "pattern": pattern_types[pattern_idx],
            "seed": np.arange(1000, 1000 + n),
        }
    )

    return df, X


# ============================================================
# T-SNE + CLUSTERING
# ============================================================

@st.cache_data
def run_analysis(
    feature_set,
    clustering_method,
    n_clusters,
):

    df, X = generate_demo_data(
        feature_set
    )

    X_scaled = StandardScaler().fit_transform(
        X
    )

    coords = TSNE(
        n_components=2,
        random_state=42,
        perplexity=34,
        init="pca",
        learning_rate="auto",
    ).fit_transform(
        X_scaled
    )

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

    df["cluster"] = model.fit_predict(
        X_scaled
    )

    df["tsne_x"] = coords[:, 0]
    df["tsne_y"] = coords[:, 1]

    return df


# ============================================================
# SYNTHETIC SQUARE INSPECTION IMAGES
# ============================================================

@st.cache_data
def make_wafer_png(
    pattern,
    seed,
    size=100,
):

    rng = np.random.default_rng(
        int(seed)
    )

    base = rng.normal(
        143,
        16,
        (size, size),
    )

    yy, xx = np.mgrid[
        0:size,
        0:size
    ]

    fine_texture = (
        5 * np.sin(xx * 0.85)
        + 4 * np.sin(yy * 0.93)
        + 3 * np.sin((xx + yy) * 0.37)
    )

    base += fine_texture

    cx = size / 2 + rng.uniform(-8, 8)
    cy = size / 2 + rng.uniform(-8, 8)

    dist = np.sqrt(
        (xx - cx) ** 2
        + (yy - cy) ** 2
    )

    base += (
        13
        * np.exp(
            -(dist ** 2)
            / (2 * (size * 0.42) ** 2)
        )
    )

    if pattern == "center":

        defect = np.exp(
            -(
                (xx - size * 0.52) ** 2
                + (yy - size * 0.50) ** 2
            )
            / (
                2 * (size * 0.14) ** 2
            )
        )

        base -= 70 * defect

    elif pattern == "edge":

        edge_band = (
            np.exp(-xx / (size * 0.08))
            + np.exp(-(size - xx) / (size * 0.08))
            + np.exp(-yy / (size * 0.08))
            + np.exp(-(size - yy) / (size * 0.08))
        )

        base -= 28 * edge_band

    elif pattern == "local":

        px = rng.uniform(
            size * 0.25,
            size * 0.75,
        )

        py = rng.uniform(
            size * 0.25,
            size * 0.75,
        )

        sigma = rng.uniform(
            size * 0.08,
            size * 0.16,
        )

        local = np.exp(
            -(
                (xx - px) ** 2
                + (yy - py) ** 2
            )
            / (2 * sigma ** 2)
        )

        base -= 65 * local

    elif pattern == "band":

        angle = rng.uniform(
            -0.5,
            0.5,
        )

        line = (
            yy
            - (
                size * 0.48
                + angle
                * (
                    xx
                    - size / 2
                )
            )
        )

        band = np.exp(
            -(line ** 2)
            / (
                2
                * (
                    size * 0.055
                ) ** 2
            )
        )

        base -= 48 * band

    elif pattern == "streak":

        for _ in range(
            rng.integers(2, 5)
        ):

            x0 = rng.uniform(
                0,
                size,
            )

            width = rng.uniform(
                1.5,
                4.5,
            )

            streak = np.exp(
                -(
                    (xx - x0) ** 2
                )
                / (
                    2
                    * width ** 2
                )
            )

            base -= (
                rng.uniform(
                    18,
                    38,
                )
                * streak
            )

    elif pattern == "texture":

        coarse = (
            12
            * np.sin(
                xx * 0.18
                + rng.uniform(
                    0,
                    np.pi,
                )
            )
            * np.sin(
                yy * 0.16
                + rng.uniform(
                    0,
                    np.pi,
                )
            )
        )

        base += coarse

    elif pattern == "clean":

        base += rng.normal(
            0,
            3,
            (size, size),
        )

    # Sparse irregular features
    for _ in range(
        rng.integers(5, 18)
    ):

        px = rng.integers(
            3,
            size - 3,
        )

        py = rng.integers(
            3,
            size - 3,
        )

        radius = rng.uniform(
            0.8,
            3.5,
        )

        spot = np.exp(
            -(
                (xx - px) ** 2
                + (yy - py) ** 2
            )
            / (
                2 * radius ** 2
            )
        )

        base -= (
            rng.uniform(
                8,
                28,
            )
            * spot
        )

    base = np.clip(
        base,
        35,
        220,
    ).astype(
        np.uint8
    )

    image = Image.fromarray(
        base,
        mode="L",
    ).convert(
        "RGB"
    )

    image = image.filter(
        ImageFilter.GaussianBlur(
            radius=0.35
        )
    )

    draw = ImageDraw.Draw(
        image
    )

    draw.rectangle(
        [
            0,
            0,
            size - 1,
            size - 1,
        ],
        outline=(
            54,
            58,
            62,
        ),
        width=2,
    )

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


def wafer_image(
    pattern,
    seed,
):

    return Image.open(
        BytesIO(
            make_wafer_png(
                pattern,
                seed,
            )
        )
    ).copy()


# ============================================================
# PLOT
# ============================================================

def build_tsne_figure(
    df,
    feature_set,
    clustering_method,
    show_cluster_outlines,
):

    cluster_colors = [
        "#009E9E",
        "#A70D35",
        "#147DA5",
        "#6A45A5",
        "#D17A0B",
        "#338A64",
        "#C23B7A",
        "#68727C",
        "#A48515",
        "#25292E",
    ]

    colors = [
        cluster_colors[
            int(cluster)
            % len(cluster_colors)
        ]
        for cluster
        in df["cluster"]
    ]

    x_span = max(
        df["tsne_x"].max()
        - df["tsne_x"].min(),
        1,
    )

    y_span = max(
        df["tsne_y"].max()
        - df["tsne_y"].min(),
        1,
    )

    # Keep current thumbnail size
    image_w = x_span * 0.024
    image_h = y_span * 0.034

    fig = go.Figure()

    # Slightly larger coloured outline behind each image
    if show_cluster_outlines:

        fig.add_trace(
            go.Scatter(
                x=df["tsne_x"],
                y=df["tsne_y"],
                mode="markers",
                marker=dict(
                    symbol="square",
                    size=24,
                    color=colors,
                    opacity=1,
                    line=dict(
                        width=0,
                    ),
                ),
                text=df["labels"],
                customdata=df["cluster"],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Cluster %{customdata}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    # Invisible selection layer
    fig.add_trace(
        go.Scatter(
            x=df["tsne_x"],
            y=df["tsne_y"],
            mode="markers",
            marker=dict(
                size=24,
                opacity=0,
            ),
            text=df["labels"],
            customdata=df["cluster"],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Cluster %{customdata}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    # Wafer thumbnails
    for _, row in df.iterrows():

        fig.add_layout_image(
            dict(
                source=wafer_image(
                    row["pattern"],
                    row["seed"],
                ),
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

    # Legend traces
    if show_cluster_outlines:

        for cluster_id in sorted(
            df["cluster"].unique()
        ):

            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(
                        symbol="square",
                        size=10,
                        color=cluster_colors[
                            int(cluster_id)
                            % len(cluster_colors)
                        ],
                    ),
                    name=f"Cluster {cluster_id}",
                )
            )

    fig.update_layout(
        height=760,

        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",

        font=dict(
            family="Inter, Helvetica Neue, Arial, sans-serif",
            color="#181A1D",
            size=12,
        ),

        # Extra space on right is deliberately reserved for legend
        margin=dict(
            l=28,
            r=150,
            t=62,
            b=30,
        ),

        # Cleaner title
        title=dict(
            text=(
                f"{feature_set}"
                f"<span style='color:#A1A7AE;'> · </span>"
                f"{clustering_method}"
            ),
            x=0.015,
            xanchor="left",
            y=0.975,
            yanchor="top",
            font=dict(
                family="Inter, Helvetica Neue, Arial, sans-serif",
                color="#181A1D",
                size=17,
                weight=600,
            ),
        ),

        # Legend is now OUTSIDE the actual plotting area
        legend=dict(
            title=dict(
                text="Cluster assignment",
                font=dict(
                    family="Inter, Helvetica Neue, Arial, sans-serif",
                    color="#59616A",
                    size=11,
                ),
            ),
            font=dict(
                family="Inter, Helvetica Neue, Arial, sans-serif",
                color="#59616A",
                size=11,
            ),
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,

            x=1.02,
            xanchor="left",
            y=1,
            yanchor="top",
        ),

        hoverlabel=dict(
            font_family="Inter",
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor="#CBD1D7",
        tickcolor="#CBD1D7",
        title="t-SNE 1",
        color="#626A73",
        ticks="outside",

        # Prevent wafers from being clipped against edges
        range=[
            df["tsne_x"].min() - x_span * 0.05,
            df["tsne_x"].max() + x_span * 0.05,
        ],
    )

    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor="#CBD1D7",
        tickcolor="#CBD1D7",
        title="t-SNE 2",
        color="#626A73",
        ticks="outside",

        range=[
            df["tsne_y"].min() - y_span * 0.06,
            df["tsne_y"].max() + y_span * 0.06,
        ],
    )

    return fig


# ============================================================
# ANALYSIS
# ============================================================

with st.spinner(
    "Computing t-SNE and clustering..."
):

    df = run_analysis(
        feature_set,
        clustering_method,
        n_clusters,
    )


# ============================================================
# MAIN LAYOUT
# ============================================================

plot_col, preview_col = st.columns(
    [3.3, 1.15],
    gap="medium",
)


# ============================================================
# PLOT CARD
# ============================================================

with plot_col:

    with st.container(
        border=True
    ):

        show_cluster_outlines = st.session_state.get(
            "show_cluster_outlines",
            True,
        )

        fig = build_tsne_figure(
            df,
            feature_set,
            clustering_method,
            show_cluster_outlines,
        )

        event = st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": False,
                "scrollZoom": False,
            },
            key=(
                f"tsne_"
                f"{feature_set}_"
                f"{clustering_method}_"
                f"{n_clusters}_"
                f"{show_cluster_outlines}"
            ),
            on_select="rerun",
            selection_mode="points",
        )


# ============================================================
# SELECTED WAFER
# ============================================================

selected_idx = 0

try:

    if event.selection.points:

        selected_idx = (
            event.selection.points[0]
            .get(
                "point_index",
                0,
            )
        )

except Exception:

    selected_idx = 0


if not (
    0
    <= selected_idx
    < len(df)
):

    selected_idx = 0


selected = df.iloc[
    selected_idx
]


# ============================================================
# PREVIEW CARD
# ============================================================

with preview_col:

    with st.container(
        border=True
    ):

        st.markdown(
            """
            <div class="section-heading">
                Wafer Preview
            </div>
            <div class="section-line"></div>
            """,
            unsafe_allow_html=True,
        )

        st.image(
            wafer_image(
                selected["pattern"],
                selected["seed"],
            ),
            use_container_width=True,
        )

        st.markdown(
            f"""
            <div class="selected-wafer">
                {selected["labels"]}
            </div>

            <div class="selected-cluster">
                Cluster {selected["cluster"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="section-heading"
                 style="margin-top:24px;">
                Cluster Counts
            </div>
            <div class="section-line"></div>
            """,
            unsafe_allow_html=True,
        )

        cluster_counts = (
            df["cluster"]
            .value_counts()
            .sort_index()
            .rename_axis(
                "Cluster"
            )
            .reset_index(
                name="Count"
            )
        )

        st.dataframe(
            cluster_counts,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            """
            <div class="section-heading"
                 style="margin-top:26px;">
                Display Options
            </div>
            <div class="section-line"></div>
            """,
            unsafe_allow_html=True,
        )

        st.toggle(
            "Show cluster outlines",
            value=True,
            key="show_cluster_outlines",
        )
