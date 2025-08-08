FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY session_name.session .  # Include Telegram session file if needed

ENV PYTHONUNBUFFERED=1

CMD gunicorn -w 2 -b 0.0.0.0:$PORT affiliate_finder:app