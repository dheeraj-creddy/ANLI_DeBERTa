FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
#COPY models/ ./models/
COPY download_model.sh .
RUN chmod +x download_model.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run
#CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["sh", "-c", "./download_model.sh && uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]