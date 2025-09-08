# CLIP From Scratch

A PyTorch implementation of CLIP (Contrastive Language-Image Pre-training) built from scratch for learning and experimentation.

## Features

- 🔥Complete CLIP implementation from scratch
- 🚀 PyTorch Lightning for scalable training
- 📊 Built-in dataset handling and preprocessing
- 📊 Pre-training and inference scripts
- Interactive Gradio interface for testing

## Installation

### Prerequisites

- Python >= 3.12
- CUDA-compatible GPU (recommended)
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

### Quick Install with uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/KHRyu8985/clip-from-scratch.git
cd clip-from-scratch

# Install dependencies with uv
uv sync
```

## Usage

### 1. Data Preparation

Download and prepare your dataset:

```bash
# Make download script executable and run
chmod +x script/download_data.sh
./script/download_data.sh
```

### 2. Training

Start CLIP pre-training:

```bash
# Using uv
uv run python script/pretrain.py
```

### 3. Inference

Run inference on trained model:

```bash
# Basic inference
uv run python script/inference.py

# Interactive Gradio interface
uv run python script/gradio_inference.py
```

### 4. Jupyter Notebooks

Explore the notebooks for dataset visualization and experimentation:

```bash
# Start Jupyter Lab
uv run jupyter lab

# Open notebooks in the notebook/ directory
```

## Project Structure

```
clip-from-scratch/
 data/                    # Dataset storage (gitignored)
 logs/                    # Training logs (gitignored)
 results/                 # Experiment results (gitignored)
 weights/                 # Model weights (gitignored)
 src/                     # Source code
    dataset.py          # Dataset handling
    ...                 # Model implementations
 script/                  # Training and inference scripts
    pretrain.py         # Pre-training script
    inference.py        # Inference script
    gradio_inference.py # Interactive interface
    download_data.sh    # Data download script
 notebook/               # Jupyter notebooks
 pyproject.toml          # Project dependencies
```