FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# OpenCV needs these runtime libraries in slim Debian images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY core ./core
COPY models ./models
COPY scripts ./scripts
COPY web ./web
COPY server.py .
COPY data/coordinate_thresholds.json ./data/coordinate_thresholds.json
COPY ["data/TLFS23 - Tamil Language Finger Spelling Image Dataset/ReadMe.txt", "./data/TLFS23 - Tamil Language Finger Spelling Image Dataset/ReadMe.txt"]
COPY ["data/TLFS23 - Tamil Language Finger Spelling Image Dataset/Refrence Image", "./data/TLFS23 - Tamil Language Finger Spelling Image Dataset/Refrence Image"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000} --log-level warning --no-access-log"]
