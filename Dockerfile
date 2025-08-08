FROM python:3.11-slim

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала обновим инструменты сборки Python
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Кэш зависимостей
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Копия исходников
COPY . /app

ENV PORT=5000
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1
ENV SEARCH_ENGINE=both
ENV CLASSIFIER_DEVICE=-1
ENV LOG_TO_FILE=false
ENV TELEGRAM_ENABLED=false

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:${PORT}", "--workers", "2", "--threads", "8", "--timeout", "120"]

