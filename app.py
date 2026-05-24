import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from scipy.sparse import load_npz

from src.collaborative_filtering import collaborative_recommendation
from src.content_based_filtering import content_recommendation
from src.hybrid_recommendations import HybridRecommenderSystem


st.set_page_config(
    page_title="Soundwave · Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

    :root {
        --bg-base:      #080e0a;
        --bg-surface:   #0d1410;
        --bg-card:      #111a14;
        --bg-card-alt:  #162019;
        --accent:       #1db954;
        --accent-dim:   #17943f;
        --accent-glow:  rgba(29,185,84,0.18);
        --teal:         #4ade80;
        --teal-glow:    rgba(74,222,128,0.10);
        --text-primary: #f0f4f1;
        --text-muted:   #7a9282;
        --text-faint:   #3a5443;
        --border:       rgba(255,255,255,0.07);
        --radius-lg:    16px;
        --radius-md:    10px;
        --radius-pill:  999px;
        --shadow-green: 0 0 32px rgba(29,185,84,0.2);
        --shadow-card:  0 4px 24px rgba(0,0,0,0.5);
    }

    /* ── Reset & Base ─────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--bg-base) !important;
        color: var(--text-primary) !important;
    }

    .stApp {
        background: var(--bg-base);
        background-image:
            radial-gradient(ellipse 80% 50% at 10% -10%, rgba(29,185,84,0.07) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 90% 100%, rgba(74,222,128,0.04) 0%, transparent 60%);
    }

    /* Hide Streamlit chrome */
    #MainMenu, header, footer { visibility: hidden; }
    .block-container {
        padding: 2rem 3rem 4rem !important;
        max-width: 1100px !important;
    }

    /* ── Masthead ─────────────────────────────────────── */
    .masthead {
        display: flex;
        align-items: flex-end;
        gap: 1.5rem;
        padding: 2.5rem 0 0.5rem;
        margin-bottom: 2.5rem;
        border-bottom: 1px solid var(--border);
    }
    .masthead-wordmark {
        font-family: 'Syne', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -1.5px;
        color: var(--text-primary);
        line-height: 1;
    }
    .masthead-wordmark span {
        color: var(--accent);
    }
    .masthead-tagline {
        font-size: 0.85rem;
        font-weight: 300;
        color: var(--text-muted);
        letter-spacing: 0.04em;
        padding-bottom: 0.35rem;
        text-transform: uppercase;
    }




    /* ── Input Panel ──────────────────────────────────── */
    .input-panel {
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 2rem 2rem 1.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .input-panel::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent), var(--teal), transparent);
    }
    .panel-label {
        font-family: 'Syne', sans-serif;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-faint);
        margin-bottom: 1.2rem;
    }

    /* ── Streamlit Widget Overrides ───────────────────── */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        color: var(--text-muted) !important;
        letter-spacing: 0.03em !important;
        text-transform: uppercase !important;
    }
    div[data-baseweb="select"] > div {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
        transition: border-color 0.2s;
    }
    div[data-baseweb="select"] > div:focus-within {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px var(--accent-glow) !important;
    }
    div[data-baseweb="popover"] {
        background: var(--bg-card-alt) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
    }
    div[data-baseweb="popover"] li:hover {
        background: var(--accent-glow) !important;
        color: var(--accent) !important;
    }

    /* Slider */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: var(--accent) !important;
        border-color: var(--accent) !important;
        box-shadow: 0 0 10px var(--accent-glow) !important;
    }
    .stSlider [data-testid="stMarkdownContainer"] p {
        color: var(--text-muted) !important;
        font-size: 0.78rem !important;
    }

    /* ── Primary Button ───────────────────────────────── */
    .stButton > button[kind="primary"],
    .stButton > button {
        background: var(--accent) !important;
        color: #0a0c12 !important;
        border: none !important;
        border-radius: var(--radius-pill) !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.04em !important;
        padding: 0.6rem 1.6rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 16px var(--accent-glow) !important;
    }
    .stButton > button:hover {
        background: #22c55e !important;
        transform: translateY(-1px) !important;
        box-shadow: var(--shadow-green) !important;
    }
    .stButton > button:active { transform: translateY(0) !important; }

    /* Secondary / play buttons */
    button[data-testid^="baseButton-secondary"],
    .play-btn > .stButton > button {
        background: transparent !important;
        border: 1px solid var(--border) !important;
        color: var(--text-muted) !important;
        border-radius: var(--radius-pill) !important;
        font-size: 0.78rem !important;
        padding: 0.45rem 0.9rem !important;
        box-shadow: none !important;
    }
    .play-btn > .stButton > button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
        background: var(--accent-glow) !important;
        transform: none !important;
    }

    /* ── Mode Selector selectbox ──────────────────────── */
    .mode-select-wrap div[data-baseweb="select"] > div {
        background: var(--bg-surface) !important;
        border: 1px solid rgba(29,185,84,0.25) !important;
        border-radius: var(--radius-pill) !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }

    /* ── Now Playing Bar ──────────────────────────────── */
    .now-playing-bar {
        position: sticky;
        top: 0.5rem;
        z-index: 100;
        background: rgba(10,12,18,0.85);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(29,185,84,0.25);
        border-radius: var(--radius-lg);
        padding: 1rem 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.6), 0 0 0 1px rgba(29,185,84,0.08);
    }
    .now-playing-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: var(--accent);
        box-shadow: 0 0 8px var(--accent);
        flex-shrink: 0;
        animation: pulse-dot 1.4s ease-in-out infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.7); }
    }
    .now-playing-meta { flex: 1; min-width: 0; }
    .now-playing-label {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--accent);
        font-weight: 600;
        margin-bottom: 0.15rem;
    }
    .now-playing-title {
        font-family: 'Syne', sans-serif;
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--text-primary);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* ── Section Header ───────────────────────────────── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin: 2rem 0 1.2rem;
    }
    .section-header-line {
        flex: 1;
        height: 1px;
        background: var(--border);
    }
    .section-header-text {
        font-family: 'Syne', sans-serif;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-faint);
    }

    /* ── Selected Song Card ───────────────────────────── */
    .selected-song-card {
        background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card-alt) 100%);
        border: 1px solid rgba(29,185,84,0.2);
        border-radius: var(--radius-lg);
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1.5rem;
        box-shadow: var(--shadow-card);
    }
    .selected-song-icon {
        width: 52px; height: 52px;
        border-radius: var(--radius-md);
        background: linear-gradient(135deg, var(--accent), var(--teal));
        display: flex; align-items: center; justify-content: center;
        font-size: 1.5rem;
        flex-shrink: 0;
    }
    .selected-song-name {
        font-family: 'Syne', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.2rem;
    }
    .selected-song-artist {
        font-size: 0.88rem;
        color: var(--text-muted);
        font-weight: 300;
    }
    .selected-pill {
        margin-left: auto;
        padding: 0.25rem 0.7rem;
        border-radius: var(--radius-pill);
        background: var(--accent-glow);
        border: 1px solid rgba(29,185,84,0.25);
        color: var(--accent);
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        flex-shrink: 0;
    }

    /* ── Recommendation Track Row ─────────────────────── */
    .track-row {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 1rem 1.4rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 1.2rem;
        transition: all 0.2s ease;
    }
    .track-row:hover {
        background: var(--bg-card-alt);
        border-color: rgba(29,185,84,0.2);
        transform: translateX(4px);
    }
    .track-num {
        font-family: 'Syne', sans-serif;
        font-size: 0.8rem;
        font-weight: 700;
        color: var(--text-faint);
        width: 24px;
        text-align: right;
        flex-shrink: 0;
    }
    .track-art {
        width: 40px; height: 40px;
        border-radius: 8px;
        background: linear-gradient(135deg, #1c2238, #252d45);
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem;
        flex-shrink: 0;
    }
    .track-info { flex: 1; min-width: 0; }
    .track-name {
        font-size: 0.92rem;
        font-weight: 500;
        color: var(--text-primary);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-bottom: 0.15rem;
    }
    .track-artist {
        font-size: 0.78rem;
        color: var(--text-muted);
        font-weight: 300;
    }
    .track-no-preview {
        font-size: 0.72rem;
        color: var(--text-faint);
        font-style: italic;
    }

    /* ── Audio Player ─────────────────────────────────── */
    audio {
        width: 100%;
        height: 36px;
        border-radius: var(--radius-pill);
        accent-color: var(--accent);
        filter: invert(0);
    }

    /* ── Algo Info Banner ─────────────────────────────── */
    .algo-banner {
        background: var(--teal-glow);
        border: 1px solid rgba(74,222,128,0.2);
        border-radius: var(--radius-md);
        padding: 0.7rem 1.2rem;
        font-size: 0.82rem;
        color: var(--teal);
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Error / Warning ──────────────────────────────── */
    .stAlert {
        border-radius: var(--radius-md) !important;
        background: rgba(239,68,68,0.08) !important;
        border: 1px solid rgba(239,68,68,0.25) !important;
    }
    div[data-testid="stNotificationContentError"] {
        color: #fca5a5 !important;
    }

    /* ── Footer ───────────────────────────────────────── */
    .footer {
        margin-top: 4rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--border);
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .footer-left {
        font-family: 'Syne', sans-serif;
        font-size: 0.78rem;
        font-weight: 700;
        color: var(--text-faint);
        letter-spacing: 0.05em;
    }
    .footer-right {
        font-size: 0.72rem;
        color: var(--text-faint);
    }

    /* ── Responsive ───────────────────────────────────── */
    @media (max-width: 768px) {
        .block-container { padding: 1.2rem 1rem 3rem !important; }
        .masthead-wordmark { font-size: 2rem; }
        .selected-song-card { flex-wrap: wrap; }
    }
</style>
""", unsafe_allow_html=True)


# ── Data Paths ─────────────────────────────────────────────────────────────────
CONTENT_DATA_PATH = "data/processed/cleaned_data.parquet"
CONTENT_TRANSFORMED_PATH = "data/models/transformed_data.npz"

COLLAB_DATA_PATH = "data/processed/collab_filtered_data.csv"
COLLAB_TRACK_IDS_PATH = "data/models/collab_track_ids.npy"
COLLAB_INTERACTION_MATRIX_PATH = "data/models/collab_interaction_matrix.npz"

HYBRID_DATA_PATH = "data/processed/collab_filtered_data.csv"
HYBRID_TRACK_IDS_PATH = "data/models/collab_track_ids.npy"
HYBRID_INTERACTION_MATRIX_PATH = "data/models/collab_interaction_matrix.npz"
HYBRID_TRANSFORMED_PATH = "data/processed/transformed_hybrid_data.npz"

# Track art emojis cycle for visual variety
TRACK_ICONS = ["🎵", "🎶", "🎸", "🎹", "🥁", "🎷", "🎺", "🎻", "🪗", "🪘"]


# ── Data Loaders ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_content_artifacts():
    songs_data = pd.read_parquet(CONTENT_DATA_PATH, columns=["track_id", "name", "artist", "spotify_preview_url"])
    transformed_data = load_npz(CONTENT_TRANSFORMED_PATH)
    return songs_data, transformed_data


@st.cache_resource
def load_collaborative_artifacts():
    songs_data = pd.read_csv(COLLAB_DATA_PATH, usecols=["track_id", "name", "artist", "spotify_preview_url"])
    track_ids = np.load(COLLAB_TRACK_IDS_PATH, allow_pickle=True)
    interaction_matrix = load_npz(COLLAB_INTERACTION_MATRIX_PATH)
    return songs_data, track_ids, interaction_matrix


@st.cache_resource
def load_hybrid_artifacts():
    songs_data = pd.read_csv(HYBRID_DATA_PATH, usecols=["track_id", "name", "artist", "spotify_preview_url"])
    track_ids = np.load(HYBRID_TRACK_IDS_PATH, allow_pickle=True)
    interaction_matrix = load_npz(HYBRID_INTERACTION_MATRIX_PATH)
    transformed_matrix = load_npz(HYBRID_TRANSFORMED_PATH)
    return songs_data, track_ids, interaction_matrix, transformed_matrix


@st.cache_resource
def build_song_options(data: pd.DataFrame):
    return (
        data[["track_id", "name", "artist"]]
        .drop_duplicates(subset=["track_id"])
        .sort_values(by=["name", "artist"])
        .reset_index(drop=True)
    )


def reset_preview_state(state_prefix: str) -> None:
    st.session_state[f"{state_prefix}_active_preview_url"] = None
    st.session_state[f"{state_prefix}_active_preview_label"] = None


# ── Shared Recommendation Renderer ────────────────────────────────────────────
def render_recommendations(recommendations: pd.DataFrame, state_prefix: str) -> None:
    active_preview_url = st.session_state.get(f"{state_prefix}_active_preview_url")
    active_preview_label = st.session_state.get(f"{state_prefix}_active_preview_label")

    # Now-playing sticky bar
    if active_preview_url:
        st.markdown(f"""
            <div class="now-playing-bar">
                <div class="now-playing-dot"></div>
                <div class="now-playing-meta">
                    <div class="now-playing-label">Now Playing</div>
                    <div class="now-playing-title">{active_preview_label}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.audio(active_preview_url, format="audio/mpeg", autoplay=True)

    # Section divider
    st.markdown("""
        <div class="section-header">
            <div class="section-header-line"></div>
            <div class="section-header-text">Recommended Tracks</div>
            <div class="section-header-line"></div>
        </div>
    """, unsafe_allow_html=True)

    for position, recommendation in enumerate(recommendations.itertuples(index=False), start=1):
        song = str(recommendation.name).title()
        artist = str(recommendation.artist).title()
        preview_url = recommendation.spotify_preview_url
        icon = TRACK_ICONS[(position - 1) % len(TRACK_ICONS)]
        is_playing = (active_preview_url == preview_url and pd.notna(preview_url) and preview_url != "")

        col_main, col_btn = st.columns([10, 2])

        with col_main:
            playing_style = "border-color: rgba(29,185,84,0.45); background: linear-gradient(135deg,#0d1410,#112018);" if is_playing else ""
            st.markdown(f"""
                <div class="track-row" style="{playing_style}">
                    <div class="track-num">{'♪' if is_playing else str(position)}</div>
                    <div class="track-art">{icon}</div>
                    <div class="track-info">
                        <div class="track-name">{song}</div>
                        <div class="track-artist">{artist}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with col_btn:
            st.write("")  # vertical alignment nudge
            if pd.notna(preview_url) and preview_url != "":
                label = "⏹ Stop" if is_playing else "▶ Play"
                btn_style = "play-btn"
                if st.button(label, key=f"{state_prefix}-play-{position}", use_container_width=True):
                    if is_playing:
                        reset_preview_state(state_prefix)
                    else:
                        st.session_state[f"{state_prefix}_active_preview_url"] = preview_url
                        st.session_state[f"{state_prefix}_active_preview_label"] = f"{song} — {artist}"
                    st.rerun()
            else:
                st.markdown('<p class="track-no-preview">No preview</p>', unsafe_allow_html=True)


# ── Mode: Content-Based ────────────────────────────────────────────────────────
def render_content_based_app() -> None:
    songs_data, transformed_data = load_content_artifacts()
    song_options = build_song_options(songs_data)
    state_prefix = "content"

    st.markdown('<div class="input-panel"><div class="panel-label">🔍 Content-Based · Audio Feature Matching</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_song = st.selectbox(
            "Song",
            song_options.index.tolist(),
            format_func=lambda idx: f"{song_options.loc[idx, 'name'].title()}  ·  {song_options.loc[idx, 'artist'].title()}",
            key="content_selected_song",
            label_visibility="collapsed",
            placeholder="Search for a song…",
        )
    with col2:
        k = st.selectbox("Results", [5, 10, 15, 20], index=1, key="content_k")

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Find Similar Tracks →", key="content_submit"):
        st.session_state.show_recommendations = True
        reset_preview_state(state_prefix)

    if st.session_state.show_recommendations:
        selected_row = song_options.loc[selected_song]
        song_name_lower = selected_row["name"].lower().strip()
        artist_name_lower = selected_row["artist"].lower().strip()

        st.markdown(f"""
            <div class="selected-song-card">
                <div class="selected-song-icon">🎵</div>
                <div>
                    <div class="selected-song-name">{selected_row['name'].title()}</div>
                    <div class="selected-song-artist">{selected_row['artist'].title()}</div>
                </div>
                <div class="selected-pill">Seed Track</div>
            </div>
        """, unsafe_allow_html=True)

        recommendations = content_recommendation(
            song_name=song_name_lower,
            artist_name=artist_name_lower,
            songs_data=songs_data,
            transformed_data=transformed_data,
            k=k,
        )
        recommendations = recommendations.loc[
            ~((recommendations["name"].str.lower() == song_name_lower)
              & (recommendations["artist"].str.lower() == artist_name_lower))
        ].head(k).reset_index(drop=True)

        render_recommendations(recommendations, state_prefix)


# ── Mode: Collaborative ────────────────────────────────────────────────────────
def render_collaborative_app() -> None:
    songs_data, track_ids, interaction_matrix = load_collaborative_artifacts()
    song_options = build_song_options(songs_data)
    state_prefix = "collab"

    st.markdown('<div class="input-panel"><div class="panel-label">👥 Collaborative Filtering · Listener Behavior</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_song = st.selectbox(
            "Song",
            song_options.index.tolist(),
            format_func=lambda idx: f"{song_options.loc[idx, 'name'].title()}  ·  {song_options.loc[idx, 'artist'].title()}",
            key="collab_selected_song",
            label_visibility="collapsed",
            placeholder="Search for a song…",
        )
    with col2:
        k = st.selectbox("Results", [5, 10, 15, 20], index=0, key="collab_k")

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Find Similar Tracks →", key="collab_submit"):
        st.session_state.show_recommendations = True
        reset_preview_state(state_prefix)

    if st.session_state.show_recommendations:
        selected_row = song_options.loc[selected_song]
        song_name_lower = selected_row["name"].lower().strip()
        artist_name_lower = selected_row["artist"].lower().strip()

        matching_song = songs_data.loc[
            (songs_data["name"] == song_name_lower) & (songs_data["artist"] == artist_name_lower)
        ]
        if matching_song.empty:
            st.error(f"Couldn't find **{selected_row['name']}** by **{selected_row['artist']}** in the collaborative dataset.")
            return

        input_track_id = matching_song.iloc[0]["track_id"]
        if input_track_id not in track_ids:
            st.error(f"**{selected_row['name']}** is not present in the collaborative interaction matrix.")
            return

        st.markdown(f"""
            <div class="selected-song-card">
                <div class="selected-song-icon">🎵</div>
                <div>
                    <div class="selected-song-name">{selected_row['name'].title()}</div>
                    <div class="selected-song-artist">{selected_row['artist'].title()}</div>
                </div>
                <div class="selected-pill">Seed Track</div>
            </div>
        """, unsafe_allow_html=True)

        try:
            recommendations = collaborative_recommendation(
                song_name=song_name_lower,
                artist_name=artist_name_lower,
                track_ids=track_ids,
                songs_data=songs_data,
                interaction_matrix=interaction_matrix,
                k=k,
            )
        except ValueError as error:
            st.error(str(error))
            return

        recommendations = recommendations.loc[
            ~((recommendations["name"].str.lower() == song_name_lower)
              & (recommendations["artist"].str.lower() == artist_name_lower))
        ].head(k).reset_index(drop=True)

        render_recommendations(recommendations, state_prefix)


# ── Mode: Hybrid ───────────────────────────────────────────────────────────────
def render_hybrid_app() -> None:
    songs_data, track_ids, interaction_matrix, transformed_matrix = load_hybrid_artifacts()
    song_options = build_song_options(songs_data)
    state_prefix = "hybrid"

    st.markdown('<div class="input-panel"><div class="panel-label">⚡ Hybrid Mode · Content + Collaborative</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3, 1, 1.4])
    with col1:
        selected_song = st.selectbox(
            "Song",
            song_options.index.tolist(),
            format_func=lambda idx: f"{song_options.loc[idx, 'name'].title()}  ·  {song_options.loc[idx, 'artist'].title()}",
            key="hybrid_selected_song",
            label_visibility="collapsed",
            placeholder="Search for a song…",
        )
    with col2:
        k = st.selectbox("Results", [5, 10, 15, 20], index=1, key="hybrid_k")
    with col3:
        weight_content = st.slider(
            "Content weight",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            key="hybrid_weight",
            help="0 = fully collaborative · 1 = fully content-based",
        )

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Find Similar Tracks →", key="hybrid_submit"):
        st.session_state.show_recommendations = True
        reset_preview_state(state_prefix)

    if st.session_state.show_recommendations:
        selected_row = song_options.loc[selected_song]
        song_name_lower = selected_row["name"].lower().strip()
        artist_name_lower = selected_row["artist"].lower().strip()

        song_match = songs_data.loc[
            (songs_data["name"] == song_name_lower) & (songs_data["artist"] == artist_name_lower)
        ]
        if song_match.empty:
            st.error(f"Couldn't find **{selected_row['name']}** by **{selected_row['artist']}** in the hybrid dataset.")
            return

        input_track_id = song_match.iloc[0]["track_id"]
        if input_track_id not in track_ids:
            st.error(f"**{selected_row['name']}** is not present in the hybrid matrices.")
            return

        st.markdown(f"""
            <div class="selected-song-card">
                <div class="selected-song-icon">🎵</div>
                <div>
                    <div class="selected-song-name">{selected_row['name'].title()}</div>
                    <div class="selected-song-artist">{selected_row['artist'].title()}</div>
                </div>
                <div class="selected-pill">Seed Track</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="algo-banner">
                ⚖️ &nbsp;
                <strong>{int(weight_content * 100)}% content-based</strong>
                &nbsp;+&nbsp;
                <strong>{int((1 - weight_content) * 100)}% collaborative</strong>
                &nbsp; blended
            </div>
        """, unsafe_allow_html=True)

        hybrid_model = HybridRecommenderSystem(
            number_of_recommendations=k,
            weight_content_based=weight_content,
        )
        recommendations = hybrid_model.give_recommendations(
            song_name=song_name_lower,
            artist_name=artist_name_lower,
            songs_data=songs_data,
            track_ids=track_ids,
            transformed_matrix=transformed_matrix,
            interaction_matrix=interaction_matrix,
        )
        recommendations = recommendations.loc[
            ~((recommendations["name"].str.lower() == song_name_lower)
              & (recommendations["artist"].str.lower() == artist_name_lower))
        ].head(k).reset_index(drop=True)

        render_recommendations(recommendations, state_prefix)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    if "show_recommendations" not in st.session_state:
        st.session_state.show_recommendations = False
    if "last_mode" not in st.session_state:
        st.session_state.last_mode = "Hybrid"

    # ── Masthead ──────────────────────────────────────────────────────────────
    st.markdown("""
        <div class="masthead">
            <div class="masthead-wordmark">Sound<span>wave</span></div>
            <div class="masthead-tagline">ML Music Discovery</div>
        </div>
    """, unsafe_allow_html=True)

    # ── Mode Selector (searchable dropdown) ──────────────────────────────────
    mode_options = ["⚡ Hybrid", "🔍 Content-based", "👥 Collaborative"]
    mode_map = {"⚡ Hybrid": "Hybrid", "🔍 Content-based": "Content-based", "👥 Collaborative": "Collaborative"}
    reverse_map = {v: k for k, v in mode_map.items()}

    st.markdown('<div class="mode-select-wrap">', unsafe_allow_html=True)
    col_select, col_spacer = st.columns([2, 3])
    with col_select:
        selected_mode_label = st.selectbox(
            "Recommendation mode",
            options=mode_options,
            index=mode_options.index(reverse_map.get(st.session_state.last_mode, "⚡ Hybrid")),
            label_visibility="collapsed",
            key="mode_selector",
        )
    st.markdown('</div>', unsafe_allow_html=True)

    recommendation_mode = mode_map[selected_mode_label]

    if st.session_state.last_mode != recommendation_mode:
        st.session_state.show_recommendations = False
        st.session_state.last_mode = recommendation_mode
        for prefix in ("hybrid", "content", "collab"):
            st.session_state.pop(f"{prefix}_active_preview_url", None)
            st.session_state.pop(f"{prefix}_active_preview_label", None)

    # ── Render selected mode ──────────────────────────────────────────────────
    if recommendation_mode == "Hybrid":
        render_hybrid_app()
    elif recommendation_mode == "Content-based":
        render_content_based_app()
    else:
        render_collaborative_app()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
        <div class="footer">
            <div class="footer-left">SOUNDWAVE</div>
            <div class="footer-right">Music Recommender · Content · Collaborative · Hybrid</div>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()