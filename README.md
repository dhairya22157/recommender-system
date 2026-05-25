# Spotify Recommender System

A hybrid music recommendation engine that combines collaborative filtering, content-based filtering, and a blended hybrid approach to suggest songs tailored to user preferences. The system leverages machine learning techniques to analyze song audio features and user listening patterns.

**Live Demo:** https://huggingface.co/spaces/Dhairya2309/spotify-recommender-hybrid

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Recommendation Techniques](#recommendation-techniques)
- [Project Structure](#project-structure)
- [MLOps Workflow](#mlops-workflow)
- [Local Setup](#local-setup)
- [Docker Setup](#docker-setup)
- [Deployment](#deployment)
- [Future Improvements](#future-improvements)

## Features

- **Three recommendation approaches** to choose from:
  - Collaborative Filtering: Find recommendations based on user listening patterns
  - Content-Based Filtering: Discover songs similar to a seed track based on audio features
  - Hybrid Mode: Blend both approaches with configurable weights

- **Audio preview playback** for recommended tracks
- **Interactive web UI** built with Streamlit
- **Fast inference** using pre-trained models and cached data
- **Containerized deployment** with Docker
- **Automated CI/CD pipeline** using GitHub Actions

## Architecture

The system is organized into three main components:

1. **Data Processing Pipeline** - Loads and processes datasets tracked with DVC
2. **Recommendation Engines** - Implements three separate recommendation algorithms
3. **Web Interface** - Streamlit-based UI for user interaction

Data flows from the pre-trained models and processed datasets directly into the recommendation engines, which compute similarities and generate ranked suggestions. The models are trained offline and versioned using DVC, allowing the inference service to remain lightweight and fast.

## Recommendation Techniques

### Collaborative Filtering

Uses the sparse user-item interaction matrix to find similar users and recommend songs that similar users have listened to. The approach doesn't require knowledge of song features; it purely relies on listening behavior patterns.

**Implementation:** Computes cosine similarity between user preference vectors in the interaction matrix.

### Content-Based Filtering

Analyzes audio features of songs (tempo, energy, danceability, etc.) to find tracks with similar characteristics to a seed song. This approach works well for discovering musically similar tracks regardless of popularity or listening trends.

**Implementation:** Uses transformed audio feature vectors and computes cosine similarity in feature space.

### Hybrid Recommendation

Combines both collaborative and content-based signals with a configurable blend weight. Users can adjust the slider from 0% (fully collaborative) to 100% (fully content-based) to tune the recommendation behavior.

**Implementation:** Weighted linear combination of collaborative and content-based similarity scores.

## Project Structure

```
.
├── app.py                          # Main Streamlit application
├── Dockerfile                      # Docker configuration
├── requirements.txt                # Python dependencies
├── entrypoint.sh                   # Container entrypoint
├── test_app.py                     # Basic tests
│
├── src/                            # Source code modules
│   ├── collaborative_filtering.py  # Collaborative filtering algorithm
│   ├── content_based_filtering.py  # Content-based filtering algorithm
│   ├── hybrid_recommendations.py   # Hybrid recommendation engine
│   ├── data_cleaning.py            # Data preprocessing utilities
│   └── transform_filtered_data.py  # Feature transformation pipeline
│
├── data/
│   ├── raw/                        # Original datasets (DVC tracked, excluded from deployment)
│   │   ├── Music Info.csv
│   │   └── User Listening History.csv
│   ├── processed/                  # Processed and cleaned datasets
│   │   ├── cleaned_data.csv
│   │   ├── collab_filtered_data.csv
│   │   └── transformed_hybrid_data.npz
│   └── models/                     # Pre-trained models and matrices (loaded at inference)
│       ├── collab_interaction_matrix.npz
│       ├── collab_track_ids.npy
│       ├── transformed_data.npz
│       └── transformer.joblib
│
├── notebooks/                      # Jupyter notebooks for exploration and analysis
│   ├── eda_spotify.ipynb
│   ├── Spotify_Collaborative_Filtering.ipynb
│   └── 02_Spotify_Content_Based_Filtering.ipynb
│
├── deploy/                         # Deployment scripts
│   └── scripts/
│       ├── build_and_push.sh       # Docker build and push script
│       └── build_and_push.bat      # Windows batch equivalent
│
└── README.md                       # This file
```

### Key Directories

**`src/`**: Contains the core recommendation algorithms. Each module handles a specific recommendation technique and can be imported independently.

**`data/`**: Organized into three subdirectories:
- `raw/`: Original datasets are tracked with DVC but excluded from the production container to reduce size
- `processed/`: Cleaned and transformed datasets used during training
- `models/`: Pre-trained models and feature matrices loaded at inference time

**`notebooks/`**: Exploration and analysis notebooks showing the development process and model evaluation.

## MLOps Workflow

The project implements a standard MLOps pipeline:

1. **Data Versioning**: Raw datasets and model artifacts are tracked using DVC (`.dvc` files committed to Git)
2. **Training**: Models are trained offline using notebooks and scripts
3. **CI Pipeline**: GitHub Actions automatically runs tests on each commit
4. **Container Build**: Docker images are built and pushed to Docker Hub on successful CI runs
5. **Deployment**: Hugging Face Spaces pulls the latest Docker image and deploys the application

```
Git Push
   ↓
GitHub Actions (Test & Build)
   ↓
Docker Image Build
   ↓
Push to Docker Hub (dhairya2309/spotify-recommender)
   ↓
Hugging Face Spaces Deployment
   ↓
Live at port 7860
```

### DVC Usage

Data and model files are tracked using DVC:

```bash
# Pull data from DVC remote
dvc pull

# Push updates to DVC remote
dvc push
```

DVC keeps `.dvc` files in Git while the actual data is stored remotely, maintaining version control without bloating the repository.

## Local Setup

### Prerequisites

- Python 3.8 or higher
- pip or conda for package management
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Dhairya2309/spotify-recommender.git
cd spotify-recommender
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Pull data and models (requires DVC):
```bash
dvc pull
```

5. Run the application:
```bash
streamlit run app.py
```

The app will start on `http://localhost:8501`

## Docker Setup

### Pull and Run

The latest Docker image is available on Docker Hub:

```bash
docker pull dhairya2309/spotify-recommender:latest
```

Run the container:

```bash
docker run -p 7860:7860 dhairya2309/spotify-recommender:latest
```

The application will be accessible at `http://localhost:7860`

### Build Locally

To build the Docker image locally:

```bash
docker build -t spotify-recommender:local .
```

Run the local image:

```bash
docker run -p 7860:7860 spotify-recommender:local
```

### Container Details

- **Base Image**: Python 3.9 slim
- **Port**: 7860 (Streamlit default)
- **Entrypoint**: Runs `streamlit run app.py --server.port 7860`
- **Size**: Optimized to exclude raw datasets; only pre-trained models and processed data are included

## Deployment

### Hugging Face Spaces

The application is deployed on Hugging Face Spaces using a Docker container. The deployment is automatic:

1. Changes are pushed to the main Git branch
2. GitHub Actions runs tests and builds the Docker image
3. The image is pushed to Docker Hub
4. Hugging Face Spaces pulls and deploys the latest image

**Live App**: https://huggingface.co/spaces/Dhairya2309/spotify-recommender-hybrid

### CI/CD Pipeline

GitHub Actions workflow automatically:
- Runs tests on each push
- Builds and tags the Docker image
- Pushes to Docker Hub with `:latest` tag
- Triggers deployment on Hugging Face Spaces

### Environment

The application uses default Streamlit configurations but listens on port 7860 in production. All data and models are pre-loaded for fast inference.

## Future Improvements

- Add user rating feedback to continuously improve collaborative filtering
- Implement deep learning models for embeddings
- Add audio feature visualization for selected tracks
- Support for multiple streaming platforms beyond Spotify
- Real-time model retraining with new listening data
- Advanced metrics dashboard for recommendation performance
- Experiment tracking with MLflow
- A/B testing framework for algorithm comparison
- Caching strategies for frequently recommended tracks
