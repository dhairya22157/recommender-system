@echo off
set USERNAME=dhairya22157
set IMAGE_NAME=spotify-recommender
set TAG=latest

echo 🔐 Logging into Docker Hub...
docker login -u %USERNAME%
if %ERRORLEVEL% neq 0 (
    echo ❌ Docker login failed. Exiting...
    exit /b %ERRORLEVEL%
)

echo 🏗️ Building Docker image: %USERNAME%/%IMAGE_NAME%:%TAG%...
docker build -t %USERNAME%/%IMAGE_NAME%:%TAG% .
if %ERRORLEVEL% neq 0 (
    echo ❌ Docker build failed. Exiting...
    exit /b %ERRORLEVEL%
)

echo 🚀 Pushing Docker image to Docker Hub...
docker push %USERNAME%/%IMAGE_NAME%:%TAG%
if %ERRORLEVEL% neq 0 (
    echo ❌ Docker push failed. Exiting...
    exit /b %ERRORLEVEL%
)

echo ✅ Successfully built and pushed %USERNAME%/%IMAGE_NAME%:%TAG% to Docker Hub!
echo You can now configure Render to pull this image directly.
