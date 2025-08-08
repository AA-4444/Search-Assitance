FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PORT=8000
EXPOSE $PORT

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:$PORT", "affiliate_finder:app"]