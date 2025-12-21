FROM python:3.11-slim

WORKDIR /app

# Installeer systeem dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Installeer Python packages
RUN pip install --no-cache-dir \
    numpy \
    librosa \
    tensorflow \
    scikit-learn \
    tqdm \
    requests \
    matplotlib \
    seaborn \
    psycopg2-binary

# Kopieer source code
COPY src/ /app/src/

# Maak directories
RUN mkdir -p /app/data/raw /app/data/models /app/logs

# Environment
ENV BIRDNET_DB=/data/birds.db
ENV PG_HOST=192.168.1.25
ENV PG_DB=emsn
ENV PG_USER=emsn
ENV PG_PASS=emsn2024

# Entry point - auto trainer
CMD ["python", "-m", "src.auto_trainer"]
