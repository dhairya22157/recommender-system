# Use a stable slim base to avoid mirror/apt issues
FROM python:3.11-slim-bookworm

ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --retries 10 --timeout 120 --prefer-binary \
        streamlit \
        numpy \
        pandas \
        scipy \
        scikit-learn \
        category-encoders \
        joblib

COPY app.py ./
COPY src/ ./src/
COPY data/processed ./data/processed
COPY data/models ./data/models

EXPOSE 8000

CMD ["streamlit", "run", "app.py", "--server.port", "8000", "--server.address", "0.0.0.0"]