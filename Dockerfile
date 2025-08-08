FROM python:3.11-slim

# системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# инструменты сборки
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# СТАВИМ torch ИЗ официального CPU-индекса PyTorch (важно!)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1+cpu

# остальные зависимости
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# исходники
COPY . /app

ENV PORT=5000
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1
ENV SEARCH_ENGINE=both
ENV CLASSIFIER_DEVICE=-1
ENV LOG_TO_FILE=false
ENV TELEGRAM_ENABLED=false
# чуть аккуратнее с памятью на Railway
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1
ENV HF_HUB_DISABLE_TELEMETRY=1

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:${PORT}", "--workers", "1", "--threads", "8", "--timeout", "120"]

