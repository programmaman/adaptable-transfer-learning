
# Adaptable Transfer Learning with GNNs

This repository provides a full experimental framework for running ablation studies, structural GNN experiments, and detailed timing/statistical analysis. All experiments and plots are reproducible and containerized.

---

## Quick Start (with Docker)

### 1. Build the Docker image

```bash
docker compose build
````

### 2. Run all experiments and analysis

```bash
docker compose up
```

This will:

* Run all benchmarks via `experiments/experiment_runner.py`
* Execute the Struct-G sweep via `experiments/struct_g_sweep.py`
* Generate tables/plots using `experiments/struct_g_analysis.py`

When done, you'll see:

```
✓ all stages finished
```

---

## Manual Mode (via Conda)

If you're not using Docker, you can still run everything locally using `conda`.

### 1. Create the environment

```bash
conda create -n structgnn python=3.12 -y
conda activate structgnn
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> If you are using GPU: Make sure `torch` is installed with CUDA support:
>
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> ```

---

## Running Experiments (Locally)

From the root of the project, run:

```bash
python run.py
```

This executes:

* `experiments/experiment_runner.py` → Runs the baseline and ablation experiments
* `experiments/struct_g_sweep.py` → Runs the Struct-G sweep (multi-mode variants)
* `experiments/struct_g_analysis.py` → Generates CSVs and LaTeX tables

---

## Output Structure

* `results/` – Output folder with JSONs, CSVs, and LaTeX
* `plots/` – High-resolution ablation and performance plots
* `graphics_and_statistics/` – Analysis scripts and formatted outputs

---

## Notes

* All experiment configuration is in `run.py`
* If your experiment module has a function like `run()` or `main()`, it will be automatically invoked

---

## Questions?

Feel free to open issues or contact the authors for clarification.

