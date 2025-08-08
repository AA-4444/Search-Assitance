# gunicorn.conf.py
timeout = 60  # Increase timeout to 60 seconds
workers = 4   # Match your logs
bind = "0.0.0.0:5002"

