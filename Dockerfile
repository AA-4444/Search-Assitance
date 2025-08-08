# --- Runtime image ---
FROM python:3.11-slim

WORKDIR /app

# Системные пакеты (минимум)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
 && rm -rf /var/lib/apt/lists/*

# Копируем зависимости
COPY requirements.txt .

# Ставим Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект (включая frontend_dist рядом с affiliate_finder.py)
COPY . .

# В Railway нужен $PORT. Если его нет (локально) — используем 5000.
ENV PORT=5000
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

# ВАЖНО: биндим gunicorn на $PORT и на ПРАВИЛЬНЫЙ модуль/приложение: affiliate_finder:app
CMD ["sh", "-c", "gunicorn affiliate_finder:app -b 0.0.0.0:${PORT} --workers 2 --threads 8 --timeout 120"]


