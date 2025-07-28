FROM python:3.9-slim

# Install system dependencies for compilation
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch CPU version first
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model download script and pre-download models
COPY download_models.py .
RUN python download_models.py

# Copy application files
COPY . .

# Create cache directory
RUN mkdir -p .cache

# Set environment variables for offline CPU-only execution
ENV CUDA_VISIBLE_DEVICES=""
ENV OMP_NUM_THREADS=4
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# Run the document intelligence pipeline
CMD ["python", "process_collections.py"]
