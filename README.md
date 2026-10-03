# Sequential Minimal Optimization (SMO) for Support Vector Machines (SVM)

A clean, self-contained Python 3 implementation of the **Sequential Minimal Optimization (SMO)** algorithm for training soft-margin Support Vector Machines, based on [John Platt's 1998 algorithm](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf) and the lecture notes & MATLAB implementation by [Prof. Ruye Wang (Harvey Mudd College)](https://pages.hmc.edu/ruye/MachineLearning/lectures/ch9/node9.html).

---

## 📖 Overview

Support Vector Machines (SVMs) formulate binary classification as a convex quadratic programming (QP) dual problem:

$$
\max_{\alpha} L(\alpha) = \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)
$$

$$
\text{subject to } \quad 0 \le \alpha_i \le C \quad (\forall i=1,\dots,N), \quad \text{and} \quad \sum_{i=1}^{N} \alpha_i y_i = 0
$$

Standard QP solvers (such as interior-point algorithms) scale on the order of $\mathcal{O}(N^3)$, making large datasets computationally expensive. The **Sequential Minimal Optimization (SMO)** algorithm decomposes this large QP problem into the smallest possible subproblems: optimizing just **two** Lagrange multipliers $(\alpha_i, \alpha_j)$ per step while holding all other multipliers fixed.

Because of the linear equality constraint:

$$
\alpha_i y_i + \alpha_j y_j = -\sum_{k \neq i, j} \alpha_k y_k = \text{constant}
$$

the two-variable subproblem admits an exact **closed-form analytical solution**, completely eliminating the need for numerical QP matrix solvers.

---

## ✨ Features

- **Zero Heavy ML Dependencies**: Pure NumPy and Matplotlib implementation (no `scikit-learn` or `libsvm` required).
- **Exact Mathematical Equivalence**: Accurately mirrors the derivations and MATLAB code from the Harvey Mudd College lecture notes.
- **Multiple Supported Kernels**:
  - **Linear Kernel**: $K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j$
  - **Gaussian (Exponential) Kernel**: $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|)$ (matching the lecture MATLAB code)
  - **Squared RBF Kernel**: $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$
  - **Polynomial Kernel**: $K(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i^T \mathbf{x}_j)^d$
- **Configurable 2D Sample Generator**:
  - Default: **60 samples** (balanced 30 positive, 30 negative).
  - CLI flag (`-n` / `--samples`) to select sample size.
  - Distributions: Linearly Separable (Gaussian clusters), XOR Distribution, and Concentric Circles.
- **Regularization Control**: Pass regularization parameter $C$ via CLI (`-c` / `--C`) to study hard vs. soft margin behaviors.
- **Visualization**: Generates decision boundaries ($f(\mathbf{x}) = 0$), margin lines ($f(\mathbf{x}) = \pm 1$), color-shaded decision regions, and highlights support vectors.
- **Dual APIs**: Standard Object-Oriented interface (`SMOSVM`) and functional interface (`smo_train`, `smo_predict`).
- **Comprehensive CLI**: Easily configurable directly from terminal.

---

## 📂 Project Structure

```text
SVM_SMO_BLOG_FINAL/
├── svm_smo.py                 # Core SMO SVM implementation, sample generator, CLI & visualization
├── demo_comparison.py         # Side-by-side replication script for Example 1 (C=1.2 vs C=0.05)
├── test_svm_smo.py            # Automated unit and integration test suite (10 tests)
├── svm_example1_comparison.png# Side-by-side comparison figure for Example 1
├── svm_linear_C1.2.png        # Linear decision boundary with C=1.2
├── svm_linear_C0.05.png       # Linear decision boundary with C=0.05
├── svm_xor_C1.2.png           # Non-linear XOR classification with Gaussian kernel
├── svm_circular.png           # Concentric rings classification with Gaussian kernel
└── README.md                  # Documentation and mathematical explanation
```

---

## 🚀 Installation & Requirements

Ensure you have Python 3.8+ installed:

```bash
pip install numpy matplotlib
```

Or using a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
```

---

## 💻 Usage

### 1. Command-Line Interface (CLI)

Run `svm_smo.py` directly from the terminal:

```bash
# 1. Default run: 60 samples, C = 1.2, linear kernel
python3 svm_smo.py -n 60 -c 1.2 --save_fig svm_linear_C1.2.png

# 2. Replicate Example 1: Soft margin with small C = 0.05 (yields more support vectors)
python3 svm_smo.py -n 60 -c 0.05 --save_fig svm_linear_C0.05.png

# 3. Replicate Example 2: Non-linear XOR classification with Gaussian kernel
python3 svm_smo.py -n 60 -c 1.2 -k g -g 0.5 -d xor --save_fig svm_xor_C1.2.png

# 4. Replicate Example 3: Concentric circles with Gaussian kernel
python3 svm_smo.py -n 60 -c 1.0 -k g -g 0.5 -d circular --save_fig svm_circular.png

# 5. Headless mode (saves figure without displaying interactive GUI window)
python3 svm_smo.py -n 100 -c 1.0 --no_plot --save_fig result.png
```

#### CLI Options Reference

| Option | Flag | Default | Description |
|---|---|---|---|
| `--samples` | `-n` | `60` | Number of synthetic samples to generate. |
| `--C` | `-c` | `1.2` | Soft-margin regularization parameter C. |
| `--kernel` | `-k` | `'l'` | Kernel type: `'l'` (linear), `'g'` (gaussian), `'p'` (polynomial), `'rbf_sq'`. |
| `--gamma` | `-g` | `0.5` | Gaussian/RBF kernel bandwidth parameter $\gamma$ (`gm` in MATLAB). |
| `--degree` | | `3` | Degree for polynomial kernel. |
| `--dataset` | `-d` | `'linearly_separable'` | Dataset distribution: `'linearly_separable'`, `'xor'`, `'circular'`. |
| `--tol` | | `1e-3` | Convergence tolerance for KKT checks and step size. |
| `--max_iter` | | `100` | Maximum passes over dataset. |
| `--seed` | | `42` | Random seed for reproducible sample generation. |
| `--save_fig` | | `'svm_smo_boundary.png'` | Output image file path for plot. |
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

# 5. Plot decision boundary, margins, and support vectors
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

In the outer loop of the SMO algorithm, each sample $i$ is checked for violation of the KKT complementary slackness conditions:

```python
if (alpha[i] < C and yE < -tol) or (alpha[i] > 0 and yE > tol):
```

#### Core Variable Definitions:

**Model Decision Function:**

$$
u_i = f(\mathbf{x}_i) = \sum_{k=1}^{N} \alpha_k y_k K(\mathbf{x}_k, \mathbf{x}_i) + b
$$

**Prediction Error:**

$$
E_i = u_i - y_i
$$

**Margin Distance Quantity:**

$$
yE = y_i E_i = y_i (u_i - y_i) = y_i u_i - y_i^2 = y_i u_i - 1
$$

Tolerance: `tol` is a small positive buffer threshold (e.g., $10^{-3}$).

---

#### The Three KKT Conditions (Eq. 130):

At the optimal solution:
1. $\alpha_i = 0 \implies y_i u_i \ge 1 \iff yE \ge 0$ (Point is outside the margin, correctly classified)
2. $0 < \alpha_i < C \implies y_i u_i = 1 \iff yE = 0$ (Point lies strictly on the margin boundary)
3. $\alpha_i = C \implies y_i u_i \le 1 \iff yE \le 0$ (Point is inside the margin or misclassified)

#### State Matrix:

| Multiplier State | Condition on $yE$ | `(alpha < C and yE < -tol)` | `(alpha > 0 and yE > tol)` | Overall Result | KKT Status |
|:---:|:---:|:---:|:---:|:---:|:---|
| $\alpha_i = 0$ | $yE \ge -\text{tol}$ | False | False | **False** | **Optimal** (outside margin) |
| $\alpha_i = 0$ | $yE < -\text{tol}$ | **True** | False | **True** | **Violated** (should be $\alpha_i > 0$) |
| $0 < \alpha_i < C$ | $\|yE\| \le \text{tol}$ | False | False | **False** | **Optimal** (on margin) |
| $0 < \alpha_i < C$ | $yE < -\text{tol}$ | **True** | False | **True** | **Violated** (should be $\alpha_i = C$) |
| $0 < \alpha_i < C$ | $yE > \text{tol}$ | False | **True** | **True** | **Violated** (should be $\alpha_i = 0$) |
| $\alpha_i = C$ | $yE \le \text{tol}$ | False | False | **False** | **Optimal** (margin violator) |
| $\alpha_i = C$ | $yE > \text{tol}$ | False | **True** | **True** | **Violated** (should be $\alpha_i < C$) |

- **First Half `(alpha[i] < C and yE < -tol)`**: A point with $yE < -\text{tol}$ is inside the margin or misclassified. By KKT, it must have $\alpha_i = C$. If $\alpha_i < C$, it violates Conditions 1 and 2.
- **Second Half `(alpha[i] > 0 and yE > tol)`**: A point with $yE > \text{tol}$ is safely beyond the margin on the correct side. By KKT, it must have $\alpha_i = 0$. If $\alpha_i > 0$, it violates Conditions 2 and 3.

---

### 2. Analytical 2-Variable Update Rules

When a violating multiplier $\alpha_i$ is found, a second multiplier $\alpha_j$ ($j \neq i$) is selected for joint optimization:

#### 1. Feasible Box Constraint Bounds $[L, H]$

Ensures $0 \le \alpha_i, \alpha_j \le C$:

- If $y_i == y_j$:

$$
L = \max(0, \alpha_i + \alpha_j - C), \quad H = \min(C, \alpha_i + \alpha_j)
$$

- If $y_i \neq y_j$:

$$
L = \max(0, \alpha_j - \alpha_i), \quad H = \min(C, C + \alpha_j - \alpha_i)
$$

---

#### 2. Second Derivative $\eta$

The second-order derivative (curvature) of the 1D objective slice along the constraint line:

$$
\eta = 2 K(\mathbf{x}_i, \mathbf{x}_j) - K(\mathbf{x}_i, \mathbf{x}_i) - K(\mathbf{x}_j, \mathbf{x}_j) = -\|\phi(\mathbf{x}_i) - \phi(\mathbf{x}_j)\|^2 \le 0
$$

---

#### 3. Unconstrained Update and Clipping of $\alpha_j$

Analytical unconstrained update:

$$
\alpha_j^{\text{new}} = \alpha_j^{\text{old}} + \frac{y_j (E_j - E_i)}{\eta}
$$

Clipping $\alpha_j$ to the feasible bounding interval $[L, H]$:

$$
\alpha_j^{\text{clipped}} = \begin{cases} 
H & \text{if } \alpha_j^{\text{new}} > H \\ 
\alpha_j^{\text{new}} & \text{if } L \le \alpha_j^{\text{new}} \le H \\ 
L & \text{if } \alpha_j^{\text{new}} < L 
\end{cases}
$$

---

#### 4. Updating $\alpha_i$

Using the linear equality constraint invariant $\Delta \alpha_i = -y_i y_j \Delta \alpha_j$:

$$
\alpha_i^{\text{new}} = \alpha_i^{\text{old}} - y_i y_j (\alpha_j^{\text{clipped}} - \alpha_j^{\text{old}})
$$

---

#### 5. Threshold / Bias Updates ($b_1, b_2$)

Compute candidate threshold updates for index $i$ and index $j$:

$$
b_1 = b - E_i - y_i (\alpha_i^{\text{new}} - \alpha_i^{\text{old}}) K(\mathbf{x}_i, \mathbf{x}_i) - y_j (\alpha_j^{\text{clipped}} - \alpha_j^{\text{old}}) K(\mathbf{x}_j, \mathbf{x}_i)
$$

$$
b_2 = b - E_j - y_i (\alpha_i^{\text{new}} - \alpha_i^{\text{old}}) K(\mathbf{x}_i, \mathbf{x}_j) - y_j (\alpha_j^{\text{clipped}} - \alpha_j^{\text{old}}) K(\mathbf{x}_j, \mathbf{x}_j)
$$

The updated threshold $b$ is selected based on KKT support vector status:

$$
b = \begin{cases} 
b_1 & \text{if } 0 < \alpha_i^{\text{new}} < C \\ 
b_2 & \text{if } 0 < \alpha_j^{\text{new}} < C \\ 
\frac{b_1 + b_2}{2} & \text{otherwise} 
\end{cases}
$$

---

#### 6. Final Average Support Vector Bias (Page 6)

After the SMO algorithm terminates, the bias offset is recomputed as the exact average across all support vectors:

$$
b = \frac{1}{N_{sv}} \sum_{k=1}^{N_{sv}} \left( y_{sv, k} - \sum_{m=1}^{N_{sv}} \alpha_{sv, m} y_{sv, m} K(\mathbf{x}_{sv, m}, \mathbf{x}_{sv, k}) \right)
$$

---

## 📊 Experimental Results

### Replicating Example 1: Effect of Regularization Parameter $C$

Run `python3 demo_comparison.py` to compare $C = 1.2$ vs. $C = 0.05$ on the same 60-sample dataset:

```bash
python3 demo_comparison.py
```

- **Large $C = 1.2$**: Imposes a heavy penalty on margin violations. The boundary is strictly determined by local points close to the margin, resulting in **3 support vectors (5.0%)**.
- **Small $C = 0.05$**: Tolerates slack and allows margin violations. A larger number of points become support vectors (**10 support vectors, 16.7%**), producing a decision line that reflects the global distribution of the clusters.

---

## 🧪 Automated Testing

Run the test suite to verify implementation correctness:

```bash
python3 test_svm_smo.py
```

Test coverage includes:
- Default sample generation (60 samples, balanced 50/50).
- Custom sample sizes (20, 100, 150).
- Dataset generators (Linearly Separable, XOR, Concentric Circles).
- Kernel evaluations (Linear, Gaussian, Polynomial).
- Linear SVM convergence on separable data.
- Comparison of $C=1.2$ vs $C=0.05$ support vector counts.
- Non-linear separation of XOR and circular datasets.
- Functional API (`smo_train`, `smo_predict`).
- Visualization figure rendering and saving.

Output:
```text
..........
----------------------------------------------------------------------
Ran 10 tests in 12.212s

OK
```

---

## 📚 References

1. **Platt, John C.** (1998). *"Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines."* Microsoft Research Technical Report MSR-TR-98-14.
2. **Wang, Ruye.** (Harvey Mudd College). *"Sequential Minimal Optimization (SMO) Algorithm."* Lecture Notes on Machine Learning, Chapter 9. [Lecture Notes Link](https://pages.hmc.edu/ruye/MachineLearning/lectures/ch9/node9.html).

---

## 📄 License

MIT License. Free for educational, research, and commercial use.
