# Neural Networks from Scratch — Iris Classification

> A step-by-step implementation of classical neural learning algorithms (Perceptron & Delta Rule) on the Iris dataset, built entirely from scratch in Python without high-level ML frameworks.

---

## Overview

This project walks through the fundamentals of neural network learning by implementing and comparing two foundational training algorithms — the **Perceptron Learning Rule** and the **Delta Rule (Gradient Descent)** — applied to the classic Iris flower classification dataset from UCI.

The project is structured as six self-contained portions, each building on the last, covering everything from data preprocessing to activation function analysis, learning rate tuning, and a final head-to-head algorithm comparison.

---

## Project Structure

```
project-root/
├── run_all.py                  # Runs all six portions sequentially
├── requirements.txt
├── src/
│   ├── portion1_data.py        # Data loading & preprocessing
│   ├── portion2_perceptron.py  # Perceptron learning rule
│   ├── portion3_delta.py       # Delta rule (gradient descent)
│   ├── portion4_activations.py # Activation function comparison
│   ├── portion5_lr_tuning.py   # Learning rate sweep
│   └── portion6_comparison.py  # Final algorithm comparison
└── outputs/                    # Auto-generated results (see below)
```

---

## Getting Started

### Prerequisites

- Python **3.8 or later**

```bash
python --version
```

### 1. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it:

| Platform      | Command                        |
|---------------|--------------------------------|
| Windows       | `venv\Scripts\activate`        |
| macOS / Linux | `source venv/bin/activate`     |

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Project

**Option A — Run everything at once:**

```bash
python run_all.py
```

**Option B — Run each portion individually (in order):**

```bash
python src/portion1_data.py
python src/portion2_perceptron.py
python src/portion3_delta.py
python src/portion4_activations.py
python src/portion5_lr_tuning.py
python src/portion6_comparison.py
```

> Always run from the **project root folder** so that relative paths to the `outputs/` directory resolve correctly.

---

## Execution Order

> **Important:** `portion1_data.py` must always run before any other script.

It generates the preprocessed `.npy` data files that all subsequent portions depend on. If any later portion fails, re-run `portion1` first to regenerate the data files, then retry the failing script.

```
portion1 → portion2 → portion3 → portion4 → portion5 → portion6
```

---

## Output Files

After a successful run, the `outputs/` folder will contain:

### Data Files

| File | Description |
|------|-------------|
| `X_train.npy` | Scaled training features |
| `X_test.npy` | Scaled test features |
| `y_train.npy` | Training labels |
| `y_test.npy` | Test labels |
| `perc_errors.npy` | Perceptron misclassification history |
| `gd_loss.npy` | Delta Rule loss history |

### Visualizations

| File | Description |
|------|-------------|
| `portion1_exploration.png` | Dataset scatter plots |
| `portion2_perceptron.png` | Perceptron training results |
| `portion3_delta.png` | Delta Rule training results |
| `portion4_activations.png` | Activation function comparison |
| `portion5_lr_tuning.png` | Learning rate sweep results |
| `portion6_comparison.png` | Final side-by-side comparison |

---

## Dataset

| Property | Details |
|----------|---------|
| **Name** | Iris Dataset |
| **Source** | UCI Machine Learning Repository |
| **URL** | https://archive.ics.uci.edu/dataset/53/iris |
| **Loader** | `sklearn.datasets.load_iris()` |

The Iris dataset contains 150 samples across 3 flower species (*Setosa*, *Versicolor*, *Virginica*), each described by 4 features: sepal length, sepal width, petal length, and petal width.

---

## What Each Portion Does

| Portion | Script | Description |
|---------|--------|-------------|
| 1 | `portion1_data.py` | Loads the Iris dataset, scales features, splits into train/test sets, and saves `.npy` files |
| 2 | `portion2_perceptron.py` | Implements the Perceptron learning rule and tracks misclassification errors per epoch |
| 3 | `portion3_delta.py` | Implements the Delta Rule using gradient descent and tracks loss convergence |
| 4 | `portion4_activations.py` | Visualizes and compares common activation functions (Step, Sigmoid, ReLU, etc.) |
| 5 | `portion5_lr_tuning.py` | Sweeps over multiple learning rates to find the optimal value |
| 6 | `portion6_comparison.py` | Produces a final side-by-side comparison of both algorithms across key metrics |

---

## License

This project is intended for educational purposes.
