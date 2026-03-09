# Use official Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the project files
COPY . .

# Create mlruns and processed data folders
RUN mkdir -p mlruns data/processed models

# Set environment variables for MLflow
ENV MLFLOW_TRACKING_URI=file:///app/mlruns
ENV PYTHONUNBUFFERED=1

# Expose ports (for FastAPI / MLflow UI if needed)
# BAD (causes your error)
EXPOSE 8000  
EXPOSE 5000 

# GOOD (separate comment lines)
# FastAPI port
EXPOSE 8000
# MLflow UI port
EXPOSE 5000
# Default command: run preprocessing, training, and evaluation
CMD ["bash", "-c", "dvc repro && mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 & uvicorn src.main:app --host 0.0.0.0 --port 8000"]