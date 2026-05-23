import pandas as pd
from scipy.sparse import csr_matrix, save_npz
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

CLEANED_DATA_PATH = "data/processed/cleaned_data.csv"
USER_LISTENING_HISTORY_DATA_PATH = "data/raw/User Listening History.csv"
FILTERED_DATA_SAVE_PATH = "data/processed/collab_filtered_data.csv"
TRACK_IDS_SAVE_PATH = "data/models/collab_track_ids.npy"
INTERACTION_MATRIX_SAVE_PATH = "data/models/collab_interaction_matrix.npz"


def filter_songs_data(songs_data: pd.DataFrame, track_ids: list, save_df_path: str) -> pd.DataFrame:
    """
    Filter the songs data for the given track ids
    """
    # filter data based on track_ids
    filtered_data = songs_data[songs_data["track_id"].isin(track_ids)]
    # sort the data by track id
    filtered_data.sort_values(by="track_id", inplace=True)
    # rest index
    filtered_data.reset_index(drop=True, inplace=True)
    # save the data
    save_pandas_data_to_csv(filtered_data, save_df_path)
    
    return filtered_data


def save_pandas_data_to_csv(data: pd.DataFrame, file_path: str) -> None:
    """
    Save the data to a csv file
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(file_path, index=False)
    
    
def save_sparse_matrix(matrix: csr_matrix, file_path: str) -> None:
    """
    Save the sparse matrix to a npz file
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    save_npz(file_path, matrix)


def create_interaction_matrix(history_data: pd.DataFrame, track_ids_save_path, save_matrix_path) -> csr_matrix:
    # make a copy of data
    df = history_data.copy()
    
    # convert the playcount column to float
    df['playcount'] = pd.to_numeric(df['playcount'], errors='coerce').fillna(0).astype(np.float64)
    
    # convert string column to categorical
    df['user_id'] = df['user_id'].astype('category')
    df['track_id'] = df['track_id'].astype('category')
    
    # Convert user_id and track_id to numeric indices
    user_mapping = df['user_id'].cat.codes
    track_mapping = df['track_id'].cat.codes
    
    # get the list of track_ids
    track_ids = df['track_id'].cat.categories.values
    
    # save the categories
    Path(track_ids_save_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(track_ids_save_path, track_ids, allow_pickle=True)
    
    # add the index columns to the dataframe
    df = df.assign(
        user_idx=user_mapping,
        track_idx=track_mapping
    )
    
    # create the interaction matrix
    interaction_matrix = df.groupby(['track_idx', 'user_idx'], as_index=False)['playcount'].sum()
    
    # get the indices to form sparse matrix
    row_indices = interaction_matrix['track_idx']
    col_indices = interaction_matrix['user_idx']
    values = interaction_matrix['playcount']
    
    # get the shape of sparse matrix
    n_tracks = len(track_ids)
    n_users = len(df['user_id'].cat.categories)
    
    # create the sparse matrix
    interaction_matrix = csr_matrix((values, (row_indices, col_indices)), shape=(n_tracks, n_users))
    
    # save the sparse matrix
    save_sparse_matrix(interaction_matrix, save_matrix_path)
    
    
def collaborative_recommendation(song_name,artist_name,track_ids,songs_data,interaction_matrix,k=5):
    # lowercase the song name
    song_name = song_name.lower()
    
    # lowercase the artist name
    artist_name = artist_name.lower()
    
    # fetch the row from songs data
    song_row = songs_data.loc[(songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)]
    if song_row.empty:
        raise ValueError(f"Song '{song_name}' by '{artist_name}' was not found in the collaborative dataset.")
   
    # track_id of input song
    input_track_id = song_row['track_id'].values.item()

    if input_track_id not in track_ids:
        raise ValueError(
            f"Song '{song_name}' by '{artist_name}' is not present in the collaborative interaction matrix."
        )
  
    # index value of track_id
    ind = np.where(track_ids == input_track_id)[0].item()
    
    # fetch the input vector
    input_array = interaction_matrix[ind]
    
    # get similarity scores
    similarity_scores = cosine_similarity(input_array, interaction_matrix)
    
    # index values of recommendations
    recommendation_indices = np.argsort(similarity_scores.ravel())[-k-1:][::-1]
    
    # get top k recommendations
    recommendation_track_ids = track_ids[recommendation_indices]
    
    # get top scores
    top_scores = np.sort(similarity_scores.ravel())[-k-1:][::-1]
    
    # get the songs from data and print
    scores_df = pd.DataFrame({"track_id":recommendation_track_ids.tolist(),
                            "score":top_scores})
    
    top_k_songs = (
                    songs_data
                    .loc[songs_data["track_id"].isin(recommendation_track_ids)]
                    .merge(scores_df,on="track_id")
                    .sort_values(by="score",ascending=False)
                    .drop(columns=["track_id","score"])
                    .reset_index(drop=True)
                    )
    
    return top_k_songs


def main():
    # load the history data
    user_data = pd.read_csv(USER_LISTENING_HISTORY_DATA_PATH)
    
    # get the unique track ids
    unique_track_ids = user_data.loc[:, "track_id"].unique()
    unique_track_ids = unique_track_ids.tolist()
    
    # filter the songs data
    songs_data = pd.read_csv(CLEANED_DATA_PATH)
    filter_songs_data(songs_data, unique_track_ids, FILTERED_DATA_SAVE_PATH)
    
    # create the interaction matrix
    create_interaction_matrix(user_data, TRACK_IDS_SAVE_PATH, INTERACTION_MATRIX_SAVE_PATH)


if __name__ == "__main__":
    main()