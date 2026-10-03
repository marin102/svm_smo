# Sequential Minimal Optimization (SMO) for Support Vector Machines (SVM)

A clean, self-contained Python 3 implementation of the **Sequential Minimal Optimization (SMO)** algorithm for training soft-margin Support Vector Machines, based on [John Platt's 1998 algorithm](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf) and the lecture notes & MATLAB implementation by [Prof. Ruye Wang (Harvey Mudd College)](https://pages.hmc.edu/ruye/MachineLearning/lectures/ch9/node9.html).

---

## 📖 Overview

Support Vector Machines (SVMs) formulate binary classification as a convex quadratic programming (QP) dual problem:

$$\max_{\boldsymbol{\alpha}} L(\boldsymbol{\alpha}) = \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$$

$$\text{subject to } 0 \le \alpha_i \le C \quad \forall i=1,\dots,N, \quad \text{and} \quad \sum_{i=1}^N \alpha_i y_i = 0$$

Generic QP solvers (e.g., interior-point methods) scale poorly with large datasets $\mathcal{O}(N^3)$. The **Sequential Minimal Optimization (SMO)** algorithm decomposes this large QP problem into the smallest possible subproblems: optimizing just **two** Lagrange multipliers $(\alpha_i, \alpha_j)$ at each step while holding all others fixed. 

Because of the linear constraint $\alpha_i y_i + \alpha_j y_j = \text{const}$, the 2-variable subproblem admits an exact **closed-form analytical solution**, completely bypassing numerical QP routines.

---

## ✨ Features

- **Pure NumPy & Matplotlib Implementation**: No dependency on external machine learning libraries (like `scikit-learn` or `libsvm`).
- **Mathematical Fidelity**: Follows the exact derivations, formulas, and MATLAB code from the Harvey Mudd College lecture notes.
- **Multiple Kernel Support**:
  - Linear Kernel: $K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j$
  - Gaussian / Exponential Kernel: $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|_2)$
  - Squared RBF Kernel: $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|_2^2)$
  - Polynomial Kernel: $K(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i^T \mathbf{x}_j)^d$
- **Flexible Data Generator**:
  - Default: **60 samples** (balanced between classes $+1$ and $-1$).
  - Configurable sample count via options flag (`-n` / `--samples`).
  - Three dataset geometries: Linearly Separable (Gaussian clusters), XOR Distribution, and Concentric Circles.
- **Configurable Regularization $C$**: Experiment with soft vs. hard margin behavior via `-c` / `--C`.
- **Publication-Ready Visualizations**: Shaded decision regions, solid decision boundary $f(\mathbf{x}) = 0$, dashed margin lines $f(\mathbf{x}) = \pm 1$, and distinctly highlighted support vectors.
- **Dual APIs**: Object-oriented scikit-learn style estimator (`SMOSVM`) and functional interface (`smo_train`, `smo_predict`).
- **Command-Line Interface (CLI)**: Full argument parser for quick experiments directly from the terminal.

---

## 📂 Project Structure

```text
SVM_SMO_BLOG_FINAL/
├── svm_smo.py                 # Core SMO SVM implementation, sample generator, CLI & visualization
├── demo_comparison.py         # Side-by-side replication script for Example 1 (C=1.2 vs C=0.05)
├── test_svm_smo.py            # Automated unit and integration test suite (10 tests)
├── svm_example1_comparison.png# Replicated comparison figure for Example 1
├── svm_linear_C1.2.png        # Linear decision boundary with C=1.2
├── svm_linear_C0.05.png       # Linear decision boundary with C=0.05
├── svm_xor_C1.2.png           # Non-linear XOR classification with Gaussian kernel
├── svm_circular.png           # Concentric rings classification with Gaussian kernel
└── README.md                  # Documentation and mathematical explanation
```

---

## 🚀 Installation & Requirements

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install numpy matplotlib
```

Alternatively, use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
```

---

## 💻 Usage

### 1. Command-Line Interface (CLI)

Run the script directly from the terminal with custom flags:

```bash
# 1. Default run: 60 samples, C = 1.2, linear kernel
python3 svm_smo.py -n 60 -c 1.2 --save_fig svm_linear_C1.2.png

# 2. Replicate Example 1: Soft-margin with small C = 0.05 (more support vectors)
python3 svm_smo.py -n 60 -c 0.05 --save_fig svm_linear_C0.05.png

# 3. Replicate Example 2: Non-linear XOR classification with Gaussian kernel
python3 svm_smo.py -n 60 -c 1.2 -k g -g 0.5 -d xor --save_fig svm_xor_C1.2.png

# 4. Replicate Example 3: Concentric circles with Gaussian kernel
python3 svm_smo.py -n 60 -c 1.0 -k g -g 0.5 -d circular --save_fig svm_circular.png

# 5. Headless mode (saves figure without opening interactive window)
python3 svm_smo.py -n 100 -c 1.0 --no_plot --save_fig result.png
```

#### CLI Options Reference

| Argument | Shorthand | Default | Description |
|---|---|---|---|
| `--samples` | `-n` | `60` | Number of synthetic samples to generate. |
| `--C` | `-c` | `1.2` | Regularization parameter $C$. |
| `--kernel` | `-k` | `'l'` | Kernel type: `'l'` (linear), `'g'` (gaussian), `'p'` (polynomial), `'rbf_sq'`. |
| `--gamma` | `-g` | `0.5` | Gaussian kernel parameter $\gamma$ (`gm` in MATLAB). |
| `--degree` | | `3` | Degree for polynomial kernel. |
| `--dataset` | `-d` | `'linearly_separable'` | Dataset type: `'linearly_separable'`, `'xor'`, `'circular'`. |
| `--tol` | | `1e-3` | Numerical tolerance for KKT checks and step size. |
| `--max_iter` | | `100` | Maximum passes over dataset before termination. |
| `--seed` | | `42` | Random seed for reproducibility. |
| `--save_fig` | | `'svm_smo_boundary.png'` | Output image file path for visualization. |
| `--no_plot` | | `False` | Run in headless mode without displaying GUI window. |

---

### 2. Python API

#### Object-Oriented (`SMOSVM`)

```python
from svm_smo import SMOSVM, generate_samples, visualize_svm_boundary

# 1. Generate 60 samples (default)
X, y = generate_samples(n_samples=60, dataset_type='linearly_separable', random_state=42)

# 2. Initialize and fit the SVM model
svm = SMOSVM(C=1.2, kernel='l', tol=1e-3, max_iter=100)
svm.fit(X, y)

# 3. Inspect learned parameters
print(f"Number of Support Vectors: {svm.nsv}")
print(f"Bias offset b: {svm.bias:.4f}")

# 4. Evaluate predictions
y_pred = svm.predict(X)
acc = (y_pred == y).mean() * 100
print(f"Training Accuracy: {acc:.2f}%")

# 5. Plot boundary and margins
visualize_svm_boundary(svm, X, y, save_path="svm_plot.png", show=True)
```

#### Functional API (`smo_train`, `smo_predict`)

```python
from svm_smo import smo_train, smo_predict, generate_samples

X, y = generate_samples(n_samples=60, dataset_type='xor', random_state=42)
model = smo_train(X, y, C=1.2, kernel='g', gamma=0.5)
predictions = smo_predict(model, X)
```

---

## 🔬 Mathematical Breakdown of the Algorithm

### 1. Karush-Kuhn-Tucker (KKT) Violation Check

In the outer loop, the algorithm scans for data points violating the KKT complementary slackness conditions using the condition:

```python
if (alpha[i] < C and yE < -tol) or (alpha[i] > 0 and yE > tol):
```

where:
- Model output: $u_i = f(\mathbf{x}_i) = \sum_{k=1}^N \alpha_k y_k K(\mathbf{x}_k, \mathbf{x}_i) + b$
- Prediction error: $E_i = u_i - y_i$
- Margin offset: $yE = y_i E_i = y_i u_i - y_i^2 = y_i u_i - 1$

At the global optimum, the soft-margin KKT conditions require:
1. $\alpha_i = 0 \implies y_i u_i \ge 1 \iff yE \ge 0$ (Point correctly outside margin)
2. $0 < \alpha_i < C \implies y_i u_i = 1 \iff yE = 0$ (Point lies strictly on margin)
3. $\alpha_i = C \implies y_i u_i \le 1 \iff yE \le 0$ (Point inside margin or misclassified)

| Condition Check | Logic | Reason for Violation |
|---|---|---|
| `(alpha[i] < C and yE < -tol)` | $yE < -\text{tol} \implies y_i u_i < 1$ | The point violates the margin. Its multiplier should be driven up to $\alpha_i = C$, but is currently $< C$. |
| `(alpha[i] > 0 and yE > tol)` | $yE > \text{tol} \implies y_i u_i > 1$ | The point is safely beyond the margin. Its multiplier should be $\alpha_i = 0$, but is currently $> 0$. |

---

### 2. Analytical 2-Variable Update

Given an index $i$ violating KKT, a second index $j \neq i$ is selected:

1. **Box Constraint Bounds $[L, H]$**:
   - If $y_i == y_j$:
     $$L = \max(0, \alpha_i + \alpha_j - C), \quad H = \min(C, \alpha_i + \alpha_j)$$
   - If $y_i \neq y_j$:
     $$L = \max(0, \alpha_j - \alpha_i), \quad H = \min(C, C + \alpha_j - \alpha_i)$$

2. **Second Derivative $\eta$**:
   $$\eta = 2 K(\mathbf{x}_i, \mathbf{x}_j) - K(\mathbf{x}_i, \mathbf{x}_i) - K(\mathbf{x}_j, \mathbf{x}_j) = -\|\phi(\mathbf{x}_i) - \phi(\mathbf{x}_j)\|^2 \le 0$$

3. **Unconstrained & Clipped $\alpha_j$**:
   $$\alpha_j^{\text{new, unclipped}} = \alpha_j^{\text{old}} + \frac{y_j (E_j - E_i)}{\eta}$$
   $$\alpha_j^{\text{new}} = \text{clip}(\alpha_j^{\text{new, unclipped}}, L, H)$$

4. **Update $\alpha_i$**:
   $$\alpha_i^{\text{new}} = \alpha_i^{\text{old}} - y_i y_j (\alpha_j^{\text{new}} - \alpha_j^{\text{old}})$$

5. **Threshold Updates ($b_1, b_2$)**:
   $$b_1 = b - E_i - y_i (\Delta \alpha_i) K_{ii} - y_j (\Delta \alpha_j) K_{ji}$$
   $$b_2 = b - Ej - y_i (\Delta \alpha_i) K_{ij} - y_j (\Delta \alpha_j) K_{jj}$$
   $$b = \begin{cases} b_1 & \text{if } 0 < \alpha_i^{\text{new}} < C \\ b_2 & \text{if } 0 < \alpha_j^{\text{new}} < C \\ \frac{b_1 + b_2}{2} & \text{otherwise} \end{cases}$$

6. **Post-Processing Final Bias (Page 6)**:
   After convergence, the bias offset is recomputed as the exact average over all identified support vectors:
   $$b = \frac{1}{N_{sv}} \sum_{k=1}^{N_{sv}} \left( y_{sv, k} - \sum_{m=1}^{N_{sv}} \alpha_{sv, m} y_{sv, m} K(\mathbf{x}_{sv, m}, \mathbf{x}_{sv, k}) \right)$$

---

## 📊 Experimental Results

### Replicating Example 1: Effect of Regularization $C$

Run `python3 demo_comparison.py` to reproduce the side-by-side comparison on 60 samples:

```bash
python3 demo_comparison.py
```

- **Large $C = 1.2$**: Enforces a strict margin penalty. Only **3 support vectors (5.0%)** define the narrow boundary.
- **Small $C = 0.05$**: Tolerates margin violations. A larger set of **10 support vectors (16.7%)** define a wider boundary that captures the global distribution of the dataset.

---

## 🧪 Testing

The repository includes a comprehensive test suite validating numerical correctness, sample counts, kernel evaluations, KKT compliance, and plotting routines:

```bash
python3 test_svm_smo.py
```

All 10 tests run in $\approx 12$ seconds:

```text
..........
----------------------------------------------------------------------
Ran 10 tests in 12.212s

OK
```

---

## 📚 References

1. **Platt, John C.** (1998). *"Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines."* Microsoft Research Technical Report MSR-TR-98-14.
2. **Wang, Ruye.** (Harvey Mudd College). *"Sequential Minimal Optimization (SMO) Algorithm."* Lecture Notes on Machine Learning, Chapter 9. [Lecture Webpage](https://pages.hmc.edu/ruye/MachineLearning/lectures/ch9/node9.html).

---

## 📄 License

MIT License. Free for academic, personal, and commercial use.
# svm_smo
