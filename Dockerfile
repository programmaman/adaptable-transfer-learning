# Use the latest NVIDIA PyTorch image (optimized for CUDA)
FROM nvcr.io/nvidia/pytorch:25.01-py3
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
ENV PYTHONPATH="/app"
CMD ["python", "experiments/gppt_experiment_runner.py"]
