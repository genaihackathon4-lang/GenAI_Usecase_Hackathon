# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (ffmpeg, etc.)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port (not strictly needed for Cloud Run, but good practice)
EXPOSE 8080

# Run FastAPI with Uvicorn, binding to Cloud Run's $PORT
CMD exec uvicorn Backend.agents.ingestion_structuring_agent:app --host 0.0.0.0 --port $PORT
