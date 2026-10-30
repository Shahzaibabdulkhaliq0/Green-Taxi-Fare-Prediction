
FROM python:3.14-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt setup.py ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy trained model
COPY models/ ./models/

# API port
EXPOSE 8000

# Start API using Gunicorn with Uvicorn workers
CMD ["gunicorn", "src.api.main:app", \
     "--worker-class", "uvicorn_worker.UvicornWorker", \
     "--workers", "2", \
     "--bind", "0.0.0.0:8000"]
