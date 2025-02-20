# Use the latest NVIDIA PyTorch image (optimized for CUDA)
FROM nvcr.io/nvidia/pytorch:25.01-py3

# Set the working directory
WORKDIR /app

# Copy only requirements first for efficient layer caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files
COPY . /app

# Keep container running by default so you can attach to it, e.g. with an IDE
CMD ["tail", "-f", "/dev/null"]
