FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/models /app/input /app/output

# Copy the YOLO model
COPY custom_yolo_model.pt /app/models/

# Copy the main application
COPY solution1a.py .

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV CUDA_VISIBLE_DEVICES=""
ENV NUMEXPR_MAX_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

# Create volumes for input and output
VOLUME ["/app/input", "/app/output"]

# Run the application
CMD ["python", "solution1a.py"]