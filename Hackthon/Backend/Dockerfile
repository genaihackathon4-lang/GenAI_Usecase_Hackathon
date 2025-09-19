FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . /app/Backend

# Expose port (Cloud Run uses $PORT)
EXPOSE 8080
ENV PORT=8080

# Run FastAPI app (shell form so $PORT expands)
CMD uvicorn Backend.agents.ingestion_structuring_agent:app --host 0.0.0.0 --port ${PORT:-8080}

