# Use the latest NVIDIA PyTorch image (optimized for CUDA)
FROM nvcr.io/nvidia/pytorch:25.01-py3

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Install Python dependencies (this installs PyTorch, torchdata, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------
# 👇 ADD THIS SECTION HERE 👇
# Install system build deps for DGL
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --branch master --recurse-submodules https://github.com/dmlc/dgl.git /tmp/dgl \
    && cd /tmp/dgl && mkdir build && cd build \
    && cmake .. -DUSE_CUDA=ON -DUSE_GRAPHBOLT=ON -DUSE_LIBXSMM=OFF \
    && make -j4\
    && pip install .. \
    && cd / && rm -rf /tmp/dgl

# Copy application code after dependencies are installed
COPY . /app

# Ensure DGL uses the correct backend by default
ENV DGLBACKEND=pytorch
ENV PYTHONPATH="/app"

# Run your application
CMD ["python", "run.py"]
