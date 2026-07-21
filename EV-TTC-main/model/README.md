# **EV-TTC: Event-Based Time-To-Collision Prediction**

This repository contains the implementation of a neural network pipeline for **Time-To-Collision (TTC)** prediction using event-based camera data. The codebase is designed for training, evaluation, and inference of TTC models, with support for data preprocessing, model training, and exporting models to ONNX format.

---

## **Folder Layout**

### **1. Core Scripts**
- **`train.py`**  
  - Main script for training and validating the TTC model using PyTorch Lightning.
  - Supports checkpointing, logging, and visualization.

- **`onnx_export.py`**  
  - Exports a trained TTC model to ONNX format for deployment or inference.
  - Includes model optimization by cleaning unnecessary inputs from the ONNX graph.

- **`util.py`**  
  - Utility functions for loss calculation, visualization, and logging.
  - Includes the `charbonnier_loss` function and `ttc_plot` for visualizing predictions.

---

### **2. Models**
- **`ttc.py`**  
  - Defines the `TTCModel` class, a PyTorch Lightning module for TTC prediction.
  - Encapsulates the model architecture, training logic, and evaluation metrics.

- **`evslim.py`**  
  - Implements the `EVSlim` model, a lightweight neural network for event-based vision tasks.
  - Includes an encoder, ASPP (Atrous Spatial Pyramid Pooling) module, and decoder.

---

### **3. Data Handling**
- **`data/ttc_dm.py`**  
  - Implements the `TTCEF_DM` data module for loading and preprocessing TTC datasets.
  - Supports data augmentation, batching, and efficient data loading.

- **`data/datawriter.py`**  
  - Custom PyTorch Lightning callback for writing model predictions to HDF5 files.
  - Stores predicted TTC values, ground truth, and masks for both training and validation sets.

---

### **4. Configuration**
- **`conf/config.yaml`**  
  - Main configuration file for the pipeline, including training, data, and optimization settings.
  - Supports Hydra for dynamic configuration management.

- **`conf/models/evslim_ttc.yaml`**  
  - Model-specific configuration for the `EVSlim` architecture.
  - Defines encoder, ASPP, and decoder parameters, as well as data augmentation settings.

---

## **How to Use**

### **1. Training**
Run train.py to train the TTC model using the specified configuration.

**Example Command**:
```bash
python train.py models=evslim_ttc exp=base data_dir=/path/to/loc
```

**Key Configuration Parameters**:
- `data_dir`: Path to the dataset directory.
- `batch_size`: Number of samples per batch.
- `max_epochs`: Maximum number of training epochs.
- `lr`: Learning rate for the optimizer.
- `precision`: Training precision (e.g., `16-mixed` for mixed precision).

---

### **2. Exporting to ONNX**
Run onnx_export.py to export a trained TTC model to ONNX format.

**Example Command**:
```bash
python onnx_export.py --input_model checkpoints/last.ckpt --output_model ttc_model.onnx
```

**Arguments**:
- `--input_model`: Path to the trained model checkpoint.
- `--output_model`: Path to save the ONNX model.
- `--input_dims`: Input dimensions for the model (default: `[1, 6, 360, 360]`).

---

### **3. Data Preprocessing**
The data module (`TTCEF_DM`) handles loading and preprocessing of TTC datasets. It supports:
- **Augmentation**: Random flips and rotations for training data.
- **Efficient Loading**: Uses PyTorch DataLoader with multi-threading.

---

---

## **Output Structure**

### **1. Training Outputs**
- **Checkpoints**: Saved in the `log_dir` directory, with filenames like `epoch_00001.ckpt`.
- **Logs**: Metrics and visualizations are logged to TensorBoard.

### **2. ONNX Model**
- Exported ONNX models are saved to the specified path (e.g., `ttc_model.onnx`).

---
