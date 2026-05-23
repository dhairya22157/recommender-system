import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from scipy.sparse import load_npz

from src.collaborative_filtering import collaborative_recommendation
from src.content_based_filtering import content_recommendation
from src.hybrid_recommendations import HybridRecommenderSystem


st.set_page_config(page_title="Spotify Recommender", page_icon="🎵", layout="wide")


CONTENT_DATA_PATH = "data/processed/cleaned_data.csv"
CONTENT_TRANSFORMED_PATH = "data/models/transformed_data.npz"

COLLAB_DATA_PATH = "data/processed/collab_filtered_data.csv"
COLLAB_TRACK_IDS_PATH = "data/models/collab_track_ids.npy"
COLLAB_INTERACTION_MATRIX_PATH = "data/models/collab_interaction_matrix.npz"

HYBRID_DATA_PATH = "data/processed/collab_filtered_data.csv"
HYBRID_TRACK_IDS_PATH = "data/models/collab_track_ids.npy"
HYBRID_INTERACTION_MATRIX_PATH = "data/models/collab_interaction_matrix.npz"
HYBRID_TRANSFORMED_PATH = "data/processed/transformed_hybrid_data.npz"


@st.cache_resource
def load_content_artifacts():
    songs_data = pd.read_csv(CONTENT_DATA_PATH, usecols=["track_id", "name", "artist", "spotify_preview_url"])
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


def render_recommendations(recommendations: pd.DataFrame, state_prefix: str) -> None:
    st.markdown("---")
    st.subheader("📻 Recommended Tracks")

    active_preview_url = st.session_state.get(f"{state_prefix}_active_preview_url")
    active_preview_label = st.session_state.get(f"{state_prefix}_active_preview_label")

    if active_preview_url:
        st.caption(f"Now playing: {active_preview_label}")
        components.html(
            f"""
            <audio controls autoplay style="width: 100%;">
                <source src="{active_preview_url}" type="audio/mpeg">
                Your browser does not support the audio element.
            </audio>
            """,
            height=70,
        )

    for position, recommendation in enumerate(recommendations.itertuples(index=False), start=1):
        song = str(recommendation.name).title()
        artist = str(recommendation.artist).title()
        preview_url = recommendation.spotify_preview_url

        with st.container():
            rec_col1, rec_col2 = st.columns([3, 1])

            with rec_col1:
                st.markdown(f"**{position}. {song}** — {artist}")

            with rec_col2:
                if pd.notna(preview_url) and preview_url != "":
                    if st.button(
                        "▶ Play preview",
                        key=f"{state_prefix}-play-{position}",
                        use_container_width=True,
                    ):
                        st.session_state[f"{state_prefix}_active_preview_url"] = preview_url
                        st.session_state[f"{state_prefix}_active_preview_label"] = f"{song} — {artist}"
                else:
                    st.text("❌ No preview")

            st.write("---")


def render_content_based_app() -> None:
    songs_data, transformed_data = load_content_artifacts()
    song_options = build_song_options(songs_data)
    state_prefix = "content"

    st.title("🎵 Spotify Recommender")
    st.caption("Content-based filtering. Pick a track and get similar songs.")

    with st.container(border=True):
        col1, col2, col3 = st.columns([3, 2, 1])

        with col1:
            selected_song = st.selectbox(
                "Choose a song",
                song_options.index.tolist(),
                format_func=lambda idx: f"{song_options.loc[idx, 'name'].title()} — {song_options.loc[idx, 'artist'].title()}",
                key="content_selected_song",
            )

        with col2:
            k = st.selectbox("How many recommendations do you want?", [5, 10, 15, 20], index=1, key="content_k")

        with col3:
            st.write("")
            st.write("")
            if st.button("🎧 Get Recommendations", use_container_width=True, key="content_submit"):
                st.session_state.show_recommendations = True
                reset_preview_state(state_prefix)

    if st.session_state.show_recommendations:
        selected_row = song_options.loc[selected_song]
        song_name_lower = selected_row["name"].lower().strip()
        artist_name_lower = selected_row["artist"].lower().strip()

        st.success(f"Getting recommendations for **{selected_row['name'].title()}** by **{selected_row['artist'].title()}**")
        st.info(f"Your song: **{selected_row['name'].title()}** by **{selected_row['artist'].title()}**", icon="🎧")

        recommendations = content_recommendation(
            song_name=song_name_lower,
            artist_name=artist_name_lower,
            songs_data=songs_data,
            transformed_data=transformed_data,
            k=k,
        )

        recommendations = recommendations.loc[
            ~(
                (recommendations["name"].str.lower() == song_name_lower)
                & (recommendations["artist"].str.lower() == artist_name_lower)
            )
        ].head(k)
        recommendations = recommendations.reset_index(drop=True)

        render_recommendations(recommendations, state_prefix)


def render_collaborative_app() -> None:
    songs_data, track_ids, interaction_matrix = load_collaborative_artifacts()
    song_options = build_song_options(songs_data)
    state_prefix = "collab"

    st.title("🎵 Spotify Recommender")
    st.caption("Collaborative filtering. Pick a track and discover songs from listening history.")

    with st.container(border=True):
        col1, col2, col3 = st.columns([3, 2, 1])

        with col1:
            selected_song = st.selectbox(
                "Choose a song",
                song_options.index.tolist(),
                format_func=lambda idx: f"{song_options.loc[idx, 'name'].title()} — {song_options.loc[idx, 'artist'].title()}",
                key="collab_selected_song",
            )

        with col2:
            k = st.selectbox("How many recommendations do you want?", [5, 10, 15, 20], index=0, key="collab_k")

        with col3:
            st.write("")
            st.write("")
            if st.button("🎧 Get Recommendations", use_container_width=True, key="collab_submit"):
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
            st.error(
                f"❌ Sorry, we couldn't find **{selected_row['name']}** by **{selected_row['artist']}** in the collaborative dataset."
            )
            return

        input_track_id = matching_song.iloc[0]["track_id"]
        if input_track_id not in track_ids:
            st.error(
                f"❌ **{selected_row['name']}** by **{selected_row['artist']}** is not present in the collaborative interaction matrix."
            )
            return

        st.success(f"Getting recommendations for **{selected_row['name'].title()}** by **{selected_row['artist'].title()}**")
        st.info(f"Your song: **{selected_row['name'].title()}** by **{selected_row['artist'].title()}**", icon="🎧")

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
            st.error(f"❌ {error}")
            return

        recommendations = recommendations.loc[
            ~(
                (recommendations["name"].str.lower() == song_name_lower)
                & (recommendations["artist"].str.lower() == artist_name_lower)
            )
        ].head(k)
        recommendations = recommendations.reset_index(drop=True)

        render_recommendations(recommendations, state_prefix)


def render_hybrid_app() -> None:
    songs_data, track_ids, interaction_matrix, transformed_matrix = load_hybrid_artifacts()
    song_options = build_song_options(songs_data)
    state_prefix = "hybrid"

    st.title("🎵 Spotify Recommender")
    st.caption("Hybrid filtering. Blend content and collaborative signals for recommendations.")

    with st.container(border=True):
        col1, col2, col3 = st.columns([3, 2, 2])

        with col1:
            selected_song = st.selectbox(
                "Choose a song",
                song_options.index.tolist(),
                format_func=lambda idx: f"{song_options.loc[idx, 'name'].title()} — {song_options.loc[idx, 'artist'].title()}",
                key="hybrid_selected_song",
            )

        with col2:
            k = st.selectbox("How many recommendations do you want?", [5, 10, 15, 20], index=1, key="hybrid_k")

        with col3:
            weight_content = st.slider(
                "Content weight",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key="hybrid_weight",
            )

        if st.button("🎧 Get Recommendations", use_container_width=True, key="hybrid_submit"):
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
            st.error(
                f"❌ Sorry, we couldn't find **{selected_row['name']}** by **{selected_row['artist']}** in the hybrid dataset."
            )
            return

        input_track_id = song_match.iloc[0]["track_id"]
        if input_track_id not in track_ids:
            st.error(f"❌ **{selected_row['name']}** by **{selected_row['artist']}** is not present in the hybrid matrices.")
            return

        st.success(f"Getting recommendations for **{selected_row['name'].title()}** by **{selected_row['artist'].title()}**")
        st.info(f"Your song: **{selected_row['name'].title()}** by **{selected_row['artist'].title()}**", icon="🎧")

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
            ~(
                (recommendations["name"].str.lower() == song_name_lower)
                & (recommendations["artist"].str.lower() == artist_name_lower)
            )
        ].head(k)
        recommendations = recommendations.reset_index(drop=True)

        render_recommendations(recommendations, state_prefix)


def main() -> None:
    if "show_recommendations" not in st.session_state:
        st.session_state.show_recommendations = False
    if "last_mode" not in st.session_state:
        st.session_state.last_mode = "Hybrid"

    st.title("🎵 Spotify Song Recommender")
    st.caption("Choose a recommendation mode. Hybrid is the default.")

    recommendation_mode = st.selectbox(
        "Recommendation type",
        ["Hybrid", "Content-based", "Collaborative"],
        index=0,
        key="recommendation_mode",
    )

    if st.session_state.last_mode != recommendation_mode:
        st.session_state.show_recommendations = False
        st.session_state.last_mode = recommendation_mode
        for prefix in ("hybrid", "content", "collab"):
            st.session_state.pop(f"{prefix}_active_preview_url", None)
            st.session_state.pop(f"{prefix}_active_preview_label", None)

    if recommendation_mode == "Hybrid":
        render_hybrid_app()
    elif recommendation_mode == "Content-based":
        render_content_based_app()
    else:
        render_collaborative_app()


if __name__ == "__main__":
    main()