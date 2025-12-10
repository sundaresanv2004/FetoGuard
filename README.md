# FetoGuard

FetoGuard is a deep learning project for Fetal Head Segmentation using U-Net.

## 📂 Project Structure
```
.
├── configs/             
│   └── config.yaml      # Configuration (Hyperparams, Paths)
├── src/
│   ├── data_loader/     # Custom Dataset and Transforms
│   ├── models/          # Model Architecture (U-Net)
│   ├── trainer/         # Training Loop & Evaluation
│   └── utils/           # Logger & Helpers
├── main.py              # Single Entry Point
├── requirements.txt     # Strict Deps (Local/Docker)
├── requirements_colab.txt # Relaxed Deps (Colab)
└── output/              # Logs & Checkpoints
```

## 🛠️ Environment Setup

### Option 1: Google Colab (Recommended for Quick Start)
1.  **Clone** this repository and **Upload Data** (`dataset.zip` -> `data/dataset`).
2.  **Install Dependencies** (uses relaxed versions to avoid conflicts):
    ```bash
    !pip install -r requirements_colab.txt
    ```

### Option 2: Local Development
1.  **Install `uv`** (fast pip alternative):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2.  **Install Dependencies** (strict versions):
    ```bash
    uv pip install -r requirements.txt
    ```

### Option 3: Docker
```bash
docker build -t fetoguard .
docker run --gpus all -it fetoguard
```

---

## 🚀 Usage

All operations are handled by `main.py`.

### 1. Training
Train the U-Net model from scratch or using pretrained weights (configured in `config.yaml`).
```bash
# Default (Train 100 epochs, batch 8)
python3 main.py --mode train

# Resume Training from Checkpoint
python3 main.py --mode train --checkpoint output/10-12-2025/12-00-00/last_model.pth

# Custom Overrides
python3 main.py --mode train --epochs 50 --batch-size 16 --lr 0.0001
```
*   Logs and checkpoints saved to: `output/dd-mm-yyyy/HH-MM-SS/`
*   **Graphs**: Loss and Dice plots are automatically saved to `output/.../graphs/` at the end of training.

### 2. Testing
Evaluate a specific model checkpoint on the **Test Set**.
```bash
python3 main.py --mode test --checkpoint output/10-12-2025/12-00-00/best_model.pth
```
*Output: Loss and Dice Score on the test set.*

### 3. Single Image Inference (Prediction)
Generate a segmentation mask for a single image. The output will be an overlay saved to `output/test/`.
```bash
python3 main.py --mode predict \
    --image-path data/dataset/test_set/005_HC.png \
    --checkpoint output/10-12-2025/12-00-00/best_model.pth
```
*Result saved to: `output/test/<filename>.png`*

### 4. Plotting (Graph Generation)
Generate Loss and Dice plots from an existing training log CSV.
```bash
python3 main.py --mode plot --csv-path output/10-12-2025/12-00-00/training_log.csv
```
*Graphs saved to: `output/10-12-2025/12-00-00/graphs/`*

---

## ⚙️ Configuration
You can modify default behaviors in `configs/config.yaml`:
*   **Data Paths**: `data.data_dir`
*   **Model**: `model.name` (`unet_pretrained` or `unet_custom`)
*   **Training**: `train.epochs`, `train.learning_rate`
*   **Inference**: Default `image_path`
