# Use Python 3.9 as the base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /code

# Install system-level dependencies required for OpenCV and TensorFlow
# These libraries are essential for image processing and GUI-less operations
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file to the working directory
COPY ./requirements.txt /code/requirements.txt

# Upgrade pip and install Python dependencies
# Using --no-cache-dir keeps the image size small
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /code/requirements.txt

# Copy all project files from the local directory to the container (/code)
COPY . .

# Grant full read/write/execute permissions to the /code directory
# This is important for YOLOv8 as it may need to write temporary files/logs
RUN chmod -R 777 /code

# Start the application using Uvicorn
# Note: We run main:app directly to avoid "ModuleNotFoundError" caused by nested paths
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]