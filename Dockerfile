FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
# Не ставим EXPOSE с жестким портом, т.к. Railway сам пробрасывает нужный порт

CMD sh -c "gunicorn -w 4 -b 0.0.0.0:$PORT affiliate_finder:app"
