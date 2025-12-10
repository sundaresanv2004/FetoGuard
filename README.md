# FetoGuard

## Environment Setup

This project supports two primary ways to set up your environment: using `uv` locally or using Docker. Both methods ensure you have access to the necessary CUDA libraries for GPU acceleration.

### Option 1: Local Development with `uv` (Recommended)

`uv` is an extremely fast Python package installer and resolver.

1.  **Install `uv`** (if not already installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Create a virtual environment**:
    ```bash
    uv venv
    ```

3.  **Activate the environment**:
    ```bash
    source .venv/bin/activate
    ```

4.  **Install dependencies**:
    ```bash
    uv pip install -r requirements.txt
    ```

5.  **Verify CUDA availability**:
    ```bash
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    ```

### Option 2: Docker Environment

The Docker setup uses `uv` inside the container for fast build times and includes NVIDIA CUDA support.

1.  **Build the image**:
    ```bash
    docker build -t fetoguard .
    ```

2.  **Run the container**:
    To enable GPU access inside the container, you must use the `--gpus all` flag.
    ```bash
    docker run --gpus all -it fetoguard
    ```

3.  **Verify Setup**:
    Inside the container, run:
    ```bash
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    ```

## Requirements

The project dependencies are listed in `requirements.txt`. Key libraries include:
- PyTorch key
- Torchvision
- Torchaudio
- Numpy, Pandas, Scikit-learn
- Matplotlib, Tqdm, Jupyter
