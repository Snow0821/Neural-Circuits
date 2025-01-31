# Neural Circuits: XNOR/NAND-Based Binary Attention Networks

This repository contains an implementation of **XNOR/NAND-based Binary Attention Networks** built using **PyTorch Lightning** and **Poetry** for efficient dependency management.

This project was developed in collaboration with **ChatGPT** to explore innovative approaches to binary attention mechanisms.

## 🚀 Features
- Fully binary attention mechanism using **XNOR/NAND logic gates**
- **PyTorch Lightning** for modular training and scalability
- **Poetry-based environment** for reproducibility
- Supports both **CPU and CUDA** installations

---

## 📦 Installation Guide

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-repo/neural-circuits.git
cd neural-circuits
```

### 2️⃣ Install dependencies

#### ✅ Basic installation (without PyTorch)
```bash
poetry install
```

#### ✅ Install PyTorch with CPU support
```bash
poetry install --with dev
```

#### ✅ Install PyTorch with CUDA support (for GPU acceleration)
1. Check your CUDA version:
   ```bash
   nvidia-smi
   ```
2. Find the matching PyTorch CUDA version:  
   🔗 [PyTorch Get Started](https://pytorch.org/get-started/locally/)
3. If `pytorch-cuda` is already added, remove it first:
   ```bash
   poetry source remove pytorch-cuda
   ```
4. Add the correct CUDA source and install (Exampled with CUDA 11.8):
   ```bash
   poetry source add pytorch-cuda -n https://download.pytorch.org/whl/cu118  # Example for CUDA 11.8
   poetry install --with cuda
   ```

---

## 🏗 Project Structure
```
neural-circuits/
│── neural_circuits/      # Main package (XNOR/NAND models & training)
│   ├── __init__.py
│   ├── model.py          # Model implementation
│   ├── train.py          # Training script
│
│── tests/                # Unit tests for model & utilities
│   ├── __init__.py
│   ├── test_model.py     # Tests for model functionality
│   ├── test_training.py  # Tests for training pipeline
│
│── pyproject.toml        # Poetry configuration
│── .gitignore            # Git ignore settings
│── README.md             # Project documentation
```

---

## 🔥 Running the Tests

We use **pytest** for unit testing. To run all tests, execute:
```bash
poetry run pytest
```

To run a specific test file:
```bash
poetry run pytest tests/test_model.py
```

---

## 🏋️‍♂️ Running the Model

To train the model using **PyTorch Lightning**:
```bash
poetry run python neural_circuits/train.py
```

If you want to test **PyTorch Lightning setup**, run:
```bash
poetry run python tests/test_lightning.py
```

---

## 📝 Citation
If you use this repository, please cite as follows:

**Author:** Choi Soon Ho  
**Title:** XNOR/NAND-based Binary Attention Networks  
**Repository:** [GitHub Link](https://github.com/Neural-Circuits)  
**Year:** 2024  

For academic citation, use:
```bibtex
@article{Choi2025,
  author    = {Choi Soon Ho},
  title     = {XNOR/NAND-based Binary Attention Networks},
  journal   = {https://github.com/Neural-Circuits},
  year      = {2025}
}
```

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 🤝 Contributing
If you would like to contribute, feel free to fork the repository and submit a pull request!

---

## 📬 Contact
For any inquiries, please contact **Choi Soon Ho** at sosaror@gmail.com.

