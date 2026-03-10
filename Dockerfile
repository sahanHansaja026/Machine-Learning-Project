# Use Python 3.10 slim (matches your local venv)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for building packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the ports
EXPOSE 8000
EXPOSE 5000

# Run FastAPI app inside src folder
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

