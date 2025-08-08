FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsqlite3-dev \
    gcc \
    g++ \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install PyTorch first to avoid conflicts
RUN pip install --no-cache-dir torch==2.4.1

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

ENV PYTHONUNBUFFERED=1

CMD gunicorn -w 2 -b 0.0.0.0:$PORT affiliate_finder:app