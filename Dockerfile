# --- Stage 1: Build environment ---
FROM python:3.11-slim

# Создаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости (если нужно для bs4, lxml и т.д.)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt


COPY . .

EXPOSE 5000


CMD ["gunicorn", "-b", "0.0.0.0:5000", "affiliate_finder:app"]



