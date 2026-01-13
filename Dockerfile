FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install PyTorch Geometric and its CUDA extensions
RUN pip install --no-cache-dir torch-geometric
RUN pip install --no-cache-dir torch-scatter     -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
RUN pip install --no-cache-dir torch-sparse      -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
RUN pip install --no-cache-dir torch-cluster     -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
RUN pip install --no-cache-dir torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.1+cu121.html

# Install your Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app
COPY /experiments/config.yaml /app/config.yaml

ENV PYTHONPATH="/app"

CMD ["python", "real_gating_ablation.py"]
