# TabTransformer & FTTransformer for Tabular Data

## Overview
This repository provides **faithful, modern PyTorch implementations** of two state-of-the-art transformer architectures for tabular data:
- **TabTransformer** ([arXiv:2012.06678](https://arxiv.org/pdf/2012.06678))
- **FTTransformer** ([arXiv:2106.11959v2](https://arxiv.org/pdf/2106.11959v2))

Both models are designed for **numerical and categorical tabular data**, support batch training/inference, GPU acceleration, robust logging, error handling, and **multi-output regression or multi-label classification**. The code is modular, readable, and ready for research or production.

---

## Features
- **TabTransformer**: Contextual embeddings for categorical features using transformer encoder blocks.
- **FTTransformer**: Feature tokenization (categorical + numerical), transformer encoder with GLU, and CLS token for prediction.
- **Multi-output support**: Works for regression (multi-target), multi-label, and standard classification.
- **Batch and GPU support**: Efficient, scalable, and production-ready.
- **Robust logging**: All major operations are logged using [loguru](https://github.com/Delgan/loguru).
- **Error handling**: All critical code paths are wrapped in try/except blocks for easy debugging.
- **Training scripts**: Use `train_tabtransformer.py` and `train_fttransformer.py` for easy model training with all parameters configurable from the command line.

---

## Installation

### 1. From PyPI (recommended for users)
```bash
pip install tabtransformer-pytorch  # (replace with actual PyPI name)
```

### 2. From Source (for development)
```bash
# Clone the repository
$ git clone <your-repo-url>
$ cd TabTransformer

# Install dependencies
$ pip install -r requirements.txt
```

---

## Usage: Training Scripts

### 1. Data Format
- **Input**: You must provide your training data as three files:
  - **Categorical features**: CSV or numpy file, shape `(num_samples, num_categorical_columns)`, integer-encoded (0 ... n_classes-1 for each column)
  - **Continuous features**: CSV or numpy file, shape `(num_samples, num_continuous_columns)`, float32
  - **Labels**: CSV or numpy file, shape `(num_samples,)` for single-output, or `(num_samples, num_outputs)` for multi-output (float32 for regression/multi-label, int64 for classification)

**Example:**
```
train_categ.npy      # shape: (N, num_categorical_columns), dtype=int64
train_cont.npy       # shape: (N, num_continuous_columns), dtype=float32
train_labels.npy     # shape: (N,) or (N, num_outputs), dtype=int64 or float32
```

- **How to save your data:**
```python
import numpy as np
np.save('train_categ.npy', x_categ)   # integer-encoded categorical
np.save('train_cont.npy', x_cont)     # float32 continuous
np.save('train_labels.npy', y)        # int64 for classification, float32 for regression/multi-label
```

### 2. Training TabTransformer
```bash
python examples/train_tabtransformer.py \
    --categ_path train_categ.npy \
    --cont_path train_cont.npy \
    --labels_path train_labels.npy \
    --categories 10 5 6 5 8 \
    --num_continuous 10 \
    --num_classes 2 \
    --dim 32 \
    --depth 6 \
    --heads 8 \
    --attn_dropout 0.1 \
    --ff_dropout 0.1 \
    --mlp_hidden_mults 4 2 \
    --mlp_act relu \
    --epochs 10 \
    --batch_size 64 \
    --lr 1e-3
```

### 3. Training FTTransformer
```bash
python examples/train_fttransformer.py \
    --categ_path train_categ.npy \
    --cont_path train_cont.npy \
    --labels_path train_labels.npy \
    --categories 10 5 6 5 8 \
    --num_continuous 10 \
    --num_classes 2 \
    --dim 32 \
    --depth 6 \
    --heads 8 \
    --attn_dropout 0.1 \
    --ff_dropout 0.1 \
    --epochs 10 \
    --batch_size 64 \
    --lr 1e-3
```

#### **All Parameters:**
- `--categ_path`: Path to categorical features file (npy or csv)
- `--cont_path`: Path to continuous features file (npy or csv)
- `--labels_path`: Path to labels file (npy or csv)
- `--categories`: List of unique values per categorical column (e.g., 10 5 6 5 8)
- `--num_continuous`: Number of continuous columns
- `--num_classes`: Number of output classes or outputs (for regression/multi-label, set to output dimension)
- `--dim`: Embedding dimension (default: 32)
- `--depth`: Number of transformer layers (default: 6)
- `--heads`: Number of attention heads (default: 8)
- `--attn_dropout`: Attention dropout (default: 0.1)
- `--ff_dropout`: Feedforward dropout (default: 0.1)
- `--mlp_hidden_mults`: Multipliers for MLP hidden layers (default: 4 2, TabTransformer only)
- `--mlp_act`: Activation function for MLP (`relu` or `selu`, default: relu, TabTransformer only)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--lr`: Learning rate

---

## Multi-Output & Multi-Label Support

### 1. Multi-Output Regression
- **Labels shape:** `(num_samples, num_outputs)` (e.g., `(N, 5)` for 5 regression targets)
- **Set** `--num_classes 5` (or your output dimension)
- **Change loss in script to:**
  ```python
  criterion = torch.nn.MSELoss()
  yb = torch.tensor(y[idx:idx+args.batch_size], dtype=torch.float32, device=device)
  ```
- **Labels dtype:** `float32`

### 2. Multi-Label Classification
- **Labels shape:** `(num_samples, num_outputs)` (e.g., `(N, 5)` for 5 binary labels)
- **Set** `--num_classes 5`
- **Change loss in script to:**
  ```python
  criterion = torch.nn.BCEWithLogitsLoss()
  yb = torch.tensor(y[idx:idx+args.batch_size], dtype=torch.float32, device=device)
  ```
- **Labels dtype:** `float32` (with values 0 or 1)

### 3. Multi-Class, Multi-Output (rare)
- Each output is a separate multi-class problem. Use a custom loss (e.g., sum of CrossEntropyLoss for each output column).

---

## API Documentation

### TabTransformer
```python
from tabtransformer.model import TabTransformer

model = TabTransformer(
    categories=(10, 5, 6, 5, 8),   # tuple: unique values per categorical column
    num_continuous=10,             # number of continuous features
    dim=32,                       # embedding dimension
    dim_out=2,                    # output dimension (e.g., num classes or outputs)
    depth=6,                      # number of transformer layers
    heads=8,                      # number of attention heads
    attn_dropout=0.1,              # attention dropout
    ff_dropout=0.1,                # feedforward dropout
    mlp_hidden_mults=(4, 2),      # MLP hidden layer multipliers
    mlp_act=nn.ReLU()             # activation function
)

# Forward pass
out = model(x_categ, x_cont)
# x_categ: (batch, num_categ), torch.LongTensor
# x_cont: (batch, num_cont), torch.FloatTensor
```

### FTTransformer
```python
from fttransformer.model import FTTransformer

model = FTTransformer(
    categories=(10, 5, 6, 5, 8),   # tuple: unique values per categorical column
    num_continuous=10,             # number of continuous features
    dim=32,                       # embedding dimension
    dim_out=2,                    # output dimension (e.g., num classes or outputs)
    depth=6,                       # number of transformer layers
    heads=8,                       # number of attention heads
    attn_dropout=0.1,              # attention dropout
    ff_dropout=0.1                 # feedforward dropout
)

# Forward pass
out = model(x_categ, x_cont)
# x_categ: (batch, num_categ), torch.LongTensor
# x_cont: (batch, num_cont), torch.FloatTensor
```

---

## Logging & Error Handling
- All major operations are logged using `loguru`.
- Errors are caught and logged with stack traces for easy debugging.
- You can control log level and output by editing `utils/logger.py`.

---

## Project Structure
```
TabTransformer/
├── tabtransformer/         # TabTransformer model
│   └── model.py
├── fttransformer/          # FTTransformer model
│   └── model.py
├── utils/                  # Utilities (logging, batching, device)
│   ├── logger.py
│   ├── device.py
│   └── batch.py
├── examples/               # Training scripts
│   ├── train_tabtransformer.py
│   └── train_fttransformer.py
├── requirements.txt
└── README.md
```

---

## References
- [TabTransformer: Tabular Data Modeling Using Contextual Embeddings (arXiv:2012.06678)](https://arxiv.org/pdf/2012.06678)
- [Revisiting Deep Learning Models for Tabular Data (FTTransformer, arXiv:2106.11959v2)](https://arxiv.org/pdf/2106.11959v2)

---

---

## Publishing to PyPI (for maintainers)

1. **Update `setup.py` and `pyproject.toml`**
   - Set the package name (e.g., `tabtransformer-pytorch`), version, author, description, etc.
   - Make sure all dependencies are listed.

2. **Build the package**
```bash
python setup.py sdist bdist_wheel
```

3. **Upload to PyPI**
   - Install [twine](https://twine.readthedocs.io/en/stable/):
     ```bash
     pip install twine
     ```
   - Upload:
     ```bash
     twine upload dist/*
     ```
   - For test uploads, use `--repository testpypi`.

4. **Users can now install via**
```bash
pip install tabtransformer-pytorch
```
---