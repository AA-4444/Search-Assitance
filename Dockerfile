FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
# PORT is set by Railway at runtime, no need to hardcode
EXPOSE 8000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:$PORT", "affiliate_finder:app"]