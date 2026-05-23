#!/bin/bash
# Exit on error
set -e

# Docker Hub Username
USERNAME="dhairya22157"
IMAGE_NAME="spotify-recommender"
TAG="latest"

echo "🔐 Logging into Docker Hub..."
docker login -u "$USERNAME"

echo "🏗️ Building Docker image: $USERNAME/$IMAGE_NAME:$TAG..."
docker build -t "$USERNAME/$IMAGE_NAME:$TAG" .

echo "🚀 Pushing Docker image to Docker Hub..."
docker push "$USERNAME/$IMAGE_NAME:$TAG"

echo "✅ Successfully built and pushed $USERNAME/$IMAGE_NAME:$TAG to Docker Hub!"
echo "You can now configure Render to pull this image directly."
