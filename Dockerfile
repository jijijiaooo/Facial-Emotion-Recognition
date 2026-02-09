# Dockerfile for Azure Container Instances / App Service
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    libboost-all-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_api.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy application code
COPY api/ /app/api/
COPY src/ /app/src/
COPY models/ /app/models/

# Download shape predictor for dlib (optional - for landmarks)
# You can add this file to your repo or download it during build
# RUN wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && \
#     bunzip2 shape_predictor_68_face_landmarks.dat.bz2

# Expose port
EXPOSE 8000

# Set environment variables
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

# Run the application with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "2", "--timeout", "120", "api.app:app"]
