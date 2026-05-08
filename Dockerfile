# Use a lightweight python base image
FROM python:3.10-slim

# Set environment variables for optimized Python execution
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install crucial system dependencies required by OpenCV and Tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install uv for blazingly fast Python package management
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies using uv.
# We explicitly point to the PyTorch cu121 index to ensure GPU-compatible wheels are downloaded,
# while keeping the base image extremely lightweight.
RUN uv pip install --system --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
