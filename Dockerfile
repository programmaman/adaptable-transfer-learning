# Use the latest NVIDIA PyTorch image (optimized for CUDA)
FROM nvcr.io/nvidia/pytorch:25.01-py3
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
COPY /experiments/config.yaml /app/config.yaml
ENV PYTHONPATH="/app"
CMD ["python", "run.py"]
