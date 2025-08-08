FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
# Не ставим EXPOSE с жестким портом, т.к. Railway сам пробрасывает нужный порт

CMD sh -c "gunicorn -w 4 -b 0.0.0.0:$PORT affiliate_finder:app"
