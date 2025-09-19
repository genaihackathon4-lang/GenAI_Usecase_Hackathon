# Use official Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code into container
COPY . .

# Expose port
EXPOSE 8080

# Run FastAPI app using sh -c to expand $PORT
CMD ["sh", "-c", "uvicorn agents.ingestion_structuring_agent:app --host 0.0.0.0 --port ${PORT:-8080}"]
