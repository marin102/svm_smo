#!/usr/bin/env python3
"""
Sequential Minimal Optimization (SMO) Algorithm for Support Vector Machines (SVM)
==================================================================================

Based on the MATLAB implementation and lecture notes by Prof. Ruye Wang (Harvey Mudd College):
https://pages.hmc.edu/ruye/MachineLearning/lectures/ch9/node9.html

This module implements:
1. The SMO algorithm for training soft-margin SVMs (both linear and non-linear kernels).
2. Synthetic 2D sample generation (default: 60 samples) with configurable dataset types.
3. Decision boundary, margin, and support vector visualization.
4. Command-line interface (CLI) with flags for sample size (-n), regularization parameter C (-c),
   kernel type (-k), and dataset type (-d).
"""

import os
import sys
import argparse
import numpy as np

# Configure matplotlib cache directory to avoid non-writable permission issues in restricted environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
import matplotlib
import matplotlib.pyplot as plt


# ==============================================================================
# 1. KERNEL FUNCTIONS
# ==============================================================================

def kernel_function(X1, X2, ktype='l', gamma=0.5, degree=3):
    """
    Computes the kernel between matrix X1 and matrix/vector X2.
    
    Matches the MATLAB implementation K(X, x) from page 6:
      - 'l' or 'linear': Linear kernel K(xi, xj) = xi^T * xj
      - 'g' or 'gaussian': Gaussian kernel K(xi, xj) = exp(-gm * ||xi - xj||)
      - 'rbf_sq': Standard squared RBF kernel K(xi, xj) = exp(-gm * ||xi - xj||^2)
      - 'p' or 'polynomial': Polynomial kernel K(xi, xj) = (xi^T * xj)^3
      
    Parameters
    ----------
    X1 : np.ndarray, shape (N, D) or (D,)
        First set of sample points.
    X2 : np.ndarray, shape (M, D) or (D,)
        Second set of sample points or single point.
    ktype : str, default='l'
        Kernel identifier: 'l' (linear), 'g' (gaussian), 'p' (polynomial), 'rbf_sq'.
    gamma : float, default=0.5
        Kernel coefficient for Gaussian / RBF kernel (corresponds to 'gm' in MATLAB).
    degree : int, default=3
        Degree for polynomial kernel.
        
    Returns
    -------
    ker : np.ndarray
        Kernel evaluations. If X1 is (N, D) and X2 is (D,), returns 1D array of shape (N,).
        If both are 1D vectors, returns a float scalar.
    """
    is_x1_1d = (np.ndim(X1) == 1)
    is_x2_1d = (np.ndim(X2) == 1)
    X1_2d = np.atleast_2d(X1)
    X2_2d = np.atleast_2d(X2)

    ktype_norm = str(ktype).lower()

    if ktype_norm in ('l', 'linear'):
        # Linear kernel: inner product between points
        # MATLAB: ker(i) = X(:, i).' * x
        K = np.dot(X1_2d, X2_2d.T)

    elif ktype_norm in ('g', 'gaussian', 'rbf'):
        # Exact Gaussian formulation from MATLAB code (line 14, page 6):
        # MATLAB: ker(i) = exp(-gm * norm(X(:, i) - x))
        # Note: norm(x - z) is the Euclidean (L2) distance
        diff = X1_2d[:, np.newaxis, :] - X2_2d[np.newaxis, :, :]  # shape: (N, M, D)
        dist = np.linalg.norm(diff, axis=2)                       # shape: (N, M)
        K = np.exp(-gamma * dist)

    elif ktype_norm in ('rbf_sq', 'gaussian_sq'):
        # Standard squared Euclidean RBF kernel: exp(-gamma * ||xi - xj||^2)
        diff = X1_2d[:, np.newaxis, :] - X2_2d[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        K = np.exp(-gamma * dist_sq)

    elif ktype_norm in ('p', 'poly', 'polynomial'):
        # Polynomial kernel from MATLAB code:
        # MATLAB: ker(i) = (X(:, i).' * x).^3
        K = np.dot(X1_2d, X2_2d.T) ** degree

    else:
        raise ValueError(f"Unknown kernel type '{ktype}'. Supported: 'l', 'g', 'p', 'rbf_sq'.")

    # If both inputs were 1D vectors, return a float scalar
    if is_x1_1d and is_x2_1d:
        return float(K[0, 0])
    # If X2 was a single 1D vector and X1 was 2D, return 1D array of shape (N,)
    if is_x2_1d:
        return K[:, 0]
    # If X1 was a single 1D vector and X2 was 2D, return 1D array of shape (M,)
    if is_x1_1d:
        return K[0, :]
    return K


# ==============================================================================
# 2. SEQUENTIAL MINIMAL OPTIMIZATION (SMO) ALGORITHM
# ==============================================================================

class SMOSVM:
    """
    Support Vector Machine trained via Sequential Minimal Optimization (SMO).
    
    Implements the 2-variable analytical decomposition algorithm described in
    Prof. Ruye Wang's lecture notes (Eq. 135 - Eq. 158).
    """

    def __init__(self, C=1.0, kernel='l', gamma=0.5, degree=3, tol=1e-3, max_iter=100):
        """
        Parameters
        ----------
        C : float, default=1.0
            Soft-margin regularization parameter controlling the trade-off
            between margin width and training error slack variables.
        kernel : str, default='l'
            Kernel type: 'l' (linear), 'g' (gaussian), 'p' (polynomial).
        gamma : float, default=0.5
            Gaussian kernel bandwidth parameter (MATLAB 'gm').
        degree : int, default=3
            Polynomial kernel degree.
        tol : float, default=1e-3
            Numerical tolerance for Karush-Kuhn-Tucker (KKT) violation checks
            and convergence step size.
        max_iter : int, default=100
            Maximum number of passes (iterations) over the training dataset.
        """
        self.C = float(C)
        self.kernel = kernel
        self.gamma = float(gamma)
        self.degree = int(degree)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

        # Trained model parameters
        self.alpha = None          # Full array of Lagrange multipliers
        self.bias = 0.0            # Offset/bias term b
        self.support_indices = []  # Indices of support vectors (alpha > 0)
        self.support_alphas = None # Lagrange multipliers for support vectors
        self.support_X = None      # Support vector coordinates
        self.support_y = None      # Support vector class labels (+1 or -1)
        self.nsv = 0               # Total number of support vectors

    def _K(self, X1, X2):
        """Helper to evaluate kernel with the configured parameters."""
        return kernel_function(X1, X2, ktype=self.kernel, gamma=self.gamma, degree=self.degree)

    def fit(self, X, y):
        """
        Train the Support Vector Machine using Sequential Minimal Optimization.
        
        The algorithm solves the dual quadratic programming problem:
            maximize   L(alpha) = sum(alpha_i) - 0.5 * sum(alpha_i * alpha_j * y_i * y_j * K(x_i, x_j))
            subject to 0 <= alpha_i <= C,  for all i=1..N
                       sum(alpha_i * y_i) = 0
                       
        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Training feature matrix with N samples and D dimensions.
        y : np.ndarray, shape (N,)
            Target class labels, must be strictly +1 or -1.
            
        Returns
        -------
        self : SMOSVM
            Fitted SVM estimator.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        N, D = X.shape
        if y.shape[0] != N:
            raise ValueError(f"Shape mismatch: X has {N} samples but y has {len(y)} labels.")

        # Ensure labels are binary {-1, +1}
        unique_labels = np.unique(y)
        if not set(unique_labels).issubset({-1.0, 1.0}):
            raise ValueError(f"SVM labels must be -1 or +1, got {unique_labels}.")

        # ----------------------------------------------------------------------
        # Initialization (Corresponds to lines in MATLAB code, Page 5):
        #   alpha = zeros(1, n);  % alpha variables initialized to zero
        #   bias = 0;             % initial bias offset initialized to zero
        #   it = 0;               % iteration counter
        # ----------------------------------------------------------------------
        alpha = np.zeros(N, dtype=np.float64)
        bias = 0.0
        it = 0

        # Outer loop: repeat until convergence or maximum iterations reached
        while it < self.max_iter:
            it += 1
            changed_alphas = 0  # Counter of updated Lagrange multipliers in this pass

            # ------------------------------------------------------------------
            # Outer loop over each sample i to check for KKT condition violations
            # ------------------------------------------------------------------
            for i in range(N):
                # Compute SVM model output and error for sample i:
                # MATLAB: Ei = sum(alpha .* y .* K(X, X(:, i))) + bias - y(i);
                # Decision function: u_i = f(x_i) = sum_k(alpha_k * y_k * K(x_k, x_i)) + bias
                # Prediction error: Ei = u_i - y_i
                K_i = self._K(X, X[i])
                Ei = np.sum(alpha * y * K_i) + bias - y[i]
                yE = Ei * y[i]

                # --------------------------------------------------------------
                # KKT Complementary Slackness Conditions Check (Eq. 130):
                #   Condition 1: alpha_i == 0   => y_i * f(x_i) >= 1  => y_i * E_i >= 0
                #   Condition 2: 0 < alpha_i < C => y_i * f(x_i) == 1 => y_i * E_i == 0
                #   Condition 3: alpha_i == C   => y_i * f(x_i) <= 1  => y_i * E_i <= 0
                #
                # A sample violates KKT conditions if:
                #   (alpha_i < C and yE < -tol)  [violates Condition 1 or 2]
                #   OR
                #   (alpha_i > 0 and yE > tol)   [violates Condition 3 or 2]
                # MATLAB: if (alpha(i)<C & yE<-tol) | (alpha(i)>0 & yE>tol)
                # --------------------------------------------------------------
                if (alpha[i] < self.C and yE < -self.tol) or (alpha[i] > 0.0 and yE > self.tol):

                    # ----------------------------------------------------------
                    # Inner loop: select second variable alpha_j (j != i) to optimize
                    # MATLAB: for j=[1:i-1, i+1:N]
                    # ----------------------------------------------------------
                    for j in range(N):
                        if j == i:
                            continue

                        # Calculate error for sample j
                        # MATLAB: Ej = sum(alpha .* y .* K(X, X(:, j))) + bias - y(j);
                        K_j = self._K(X, X[j])
                        Ej = np.sum(alpha * y * K_j) + bias - y[j]

                        # Save previous alpha values before update
                        ai = alpha[i]
                        aj = alpha[j]

                        # ------------------------------------------------------
                        # Compute feasible lower (L) and upper (H) bounds for alpha_j
                        # derived from box constraints: 0 <= alpha_i, alpha_j <= C
                        # and linear equality constraint: alpha_i*y_i + alpha_j*y_j = delta
                        #
                        # Case 1: y_i == y_j (s = y_i*y_j = +1, Eq. 152)
                        #   L = max(0, alpha_i + alpha_j - C)
                        #   H = min(C, alpha_i + alpha_j)
                        #
                        # Case 2: y_i != y_j (s = y_i*y_j = -1, Eq. 153)
                        #   L = max(0, alpha_j - alpha_i)
                        #   H = min(C, C + alpha_j - alpha_i)
                        # ------------------------------------------------------
                        if y[i] == y[j]:
                            L = max(0.0, alpha[i] + alpha[j] - self.C)
                            H = min(self.C, alpha[i] + alpha[j])
                        else:
                            L = max(0.0, alpha[j] - alpha[i])
                            H = min(self.C, self.C + alpha[j] - alpha[i])

                        # If feasible line segment collapses to a single point, no optimization is possible
                        if L >= H:
                            continue

                        # ------------------------------------------------------
                        # Compute second derivative eta of objective along constraint line (Eq. 146):
                        #   eta = 2*K(x_j, x_i) - K(x_i, x_i) - K(x_j, x_j) = -||phi(x_i) - phi(x_j)||^2 <= 0
                        # MATLAB: eta = 2*K(X(:,j),X(:,i)) - K(X(:,i),X(:,i)) - K(X(:,j),X(:,j));
                        # ------------------------------------------------------
                        K_ii = self._K(X[i], X[i])
                        K_jj = self._K(X[j], X[j])
                        K_ij = self._K(X[i], X[j])  # Symmetric: K(xi, xj) == K(xj, xi)

                        eta = 2.0 * K_ij - K_ii - K_jj

                        # For Mercer kernels, eta must be negative. If eta >= 0, skip to next pair
                        if eta >= -1e-12:
                            continue

                        # ------------------------------------------------------
                        # Analytical update of alpha_j along the 1D objective (Eq. 151):
                        #   alpha_j_new = alpha_j_old + y_j * (Ej - Ei) / eta
                        # MATLAB: alpha(j) = alpha(j) + y(j)*(Ej - Ei)/eta;
                        # ------------------------------------------------------
                        alpha_j_new = alpha[j] + y[j] * (Ej - Ei) / eta

                        # ------------------------------------------------------
                        # Clip alpha_j to the feasible bounding interval [L, H] (Eq. 154):
                        # MATLAB:
                        #   if alpha(j) > H, alpha(j) = H;
                        #   elseif alpha(j) < L, alpha(j) = L;
                        # ------------------------------------------------------
                        if alpha_j_new > H:
                            alpha_j_new = H
                        elif alpha_j_new < L:
                            alpha_j_new = L

                        # If change in alpha_j is negligible, skip this pair
                        # MATLAB: if norm(alpha(j) - aj) < tol, continue
                        if abs(alpha_j_new - aj) < self.tol:
                            continue

                        # Assign clipped alpha_j
                        alpha[j] = alpha_j_new

                        # ------------------------------------------------------
                        # Update alpha_i using the equality constraint invariant (Eq. 139 & 155):
                        #   Delta(alpha_i) = - s * Delta(alpha_j) = - y_i * y_j * (alpha_j_new - alpha_j_old)
                        # MATLAB: alpha(i) = alpha(i) - y(i)*y(j)*(alpha(j) - aj);
                        # ------------------------------------------------------
                        alpha[i] = alpha[i] - y[i] * y[j] * (alpha[j] - aj)

                        # ------------------------------------------------------
                        # Update the bias/threshold term b (Eq. 157 & 158):
                        #   b1 = bias - Ei - y_i*(alpha_i - ai)*K_ii - y_j*(alpha_j - aj)*K_ji
                        #   b2 = bias - Ej - y_i*(alpha_i - ai)*K_ij - y_j*(alpha_j - aj)*K_jj
                        # MATLAB:
                        #   bi = bias - Ei - y(i)*(alpha(i)-ai)*K(X(:,i),X(:,i)) - y(j)*(alpha(j)-aj)*K(X(:,j),X(:,i));
                        #   bj = bias - Ej - y(i)*(alpha(i)-ai)*K(X(:,i),X(:,j)) - y(j)*(alpha(j)-aj)*K(X(:,j),X(:,j));
                        # ------------------------------------------------------
                        bi = bias - Ei - y[i] * (alpha[i] - ai) * K_ii - y[j] * (alpha[j] - aj) * K_ij
                        bj = bias - Ej - y[i] * (alpha[i] - ai) * K_ij - y[j] * (alpha[j] - aj) * K_jj

                        # Choose new bias according to KKT threshold conditions:
                        # If alpha_i is on the margin (0 < alpha_i < C), bi is valid.
                        # If alpha_j is on the margin (0 < alpha_j < C), bj is valid.
                        # Otherwise, take their midpoint average.
                        if 0.0 < alpha[i] < self.C:
                            bias = bi
                        elif 0.0 < alpha[j] < self.C:
                            bias = bj
                        else:
                            bias = (bi + bj) / 2.0

                        changed_alphas += 1

            # Termination condition: if no alphas were changed across an entire pass, the model has converged!
            # MATLAB: if changed_alphas == 0, break; end
            if changed_alphas == 0:
                break

        # ----------------------------------------------------------------------
        # Post-processing: Extract Support Vectors and Finalize Bias
        #
        # Corresponds to page 5 bottom & page 6 top:
        #   I = find(alpha ~= 0);  % indices of non-zero alphas
        #   alpha = alpha(I);
        #   Xsv = X(:, I);
        #   ysv = y(I);
        #   nsv = length(ysv);
        #
        # Recomputing the average bias offset across support vectors (page 6):
        #   bias = 0;
        #   for i = 1:nsv
        #       bias = bias + (ysv(i) - sum(ysv .* alpha .* K(Xsv, Xsv(:, i))));
        #   end
        #   bias = bias / nsv;
        # ----------------------------------------------------------------------
        sv_mask = (alpha > 1e-5)
        self.support_indices = np.where(sv_mask)[0]
        self.alpha = alpha
        self.support_alphas = alpha[sv_mask]
        self.support_X = X[sv_mask]
        self.support_y = y[sv_mask]
        self.nsv = len(self.support_indices)

        if self.nsv > 0:
            # Recompute exact average bias offset across support vectors (Page 6)
            bias_sum = 0.0
            for k in range(self.nsv):
                K_sv_k = self._K(self.support_X, self.support_X[k])
                margin_k = np.sum(self.support_alphas * self.support_y * K_sv_k)
                bias_sum += (self.support_y[k] - margin_k)
            self.bias = bias_sum / self.nsv
        else:
            self.bias = bias

        return self

    def decision_function(self, X):
        """
        Compute decision values f(x) for unlabeled sample points (Page 6).
        
        MATLAB:
          y = sum(alpha .* ysv .* K(Xsv, x)) + bias;
          
        Parameters
        ----------
        X : np.ndarray, shape (M, D) or (D,)
            Evaluation points.
            
        Returns
        -------
        scores : np.ndarray, shape (M,) or float
            Real-valued SVM margins.
        """
        X = np.asarray(X, dtype=np.float64)
        if self.nsv == 0:
            return np.zeros(len(X) if X.ndim > 1 else 1)

        is_1d = (X.ndim == 1)
        X_2d = np.atleast_2d(X)

        # Compute kernel between all support vectors and test points: shape (nsv, M)
        K_sv_X = self._K(self.support_X, X_2d)

        # Multiply by support vector weights (alpha * y)
        weights = (self.support_alphas * self.support_y)[:, np.newaxis]
        scores = np.sum(weights * K_sv_X, axis=0) + self.bias

        return float(scores[0]) if is_1d else scores

    def predict(self, X):
        """
        Predict binary class labels {-1, +1} for input samples.
        
        Parameters
        ----------
        X : np.ndarray, shape (M, D)
            Test features.
            
        Returns
        -------
        y_pred : np.ndarray, shape (M,)
            Predicted labels in {-1.0, +1.0}.
        """
        scores = self.decision_function(X)
        if np.isscalar(scores):
            return 1.0 if scores >= 0 else -1.0
        return np.where(scores >= 0, 1.0, -1.0)


def smo_train(X, y, C=1.0, kernel='l', gamma=0.5, degree=3, tol=1e-3, max_iter=100):
    """
    Functional interface for training an SVM using the Sequential Minimal Optimization (SMO) algorithm.
    
    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Training features.
    y : np.ndarray, shape (N,)
        Binary labels {-1, +1}.
    C : float, default=1.0
        Regularization parameter.
    kernel : str, default='l'
        Kernel type: 'l' (linear), 'g' (gaussian), 'p' (polynomial).
    gamma : float, default=0.5
        Gaussian bandwidth parameter (MATLAB 'gm').
    degree : int, default=3
        Polynomial kernel degree.
    tol : float, default=1e-3
        Convergence tolerance.
    max_iter : int, default=100
        Maximum passes over dataset.
        
    Returns
    -------
    model : SMOSVM
        Fitted SVM model.
    """
    model = SMOSVM(C=C, kernel=kernel, gamma=gamma, degree=degree, tol=tol, max_iter=max_iter)
    return model.fit(X, y)


def smo_predict(model, X):
    """
    Functional interface for predicting class labels {-1, +1} with a trained SMOSVM model.
    
    Parameters
    ----------
    model : SMOSVM
        Fitted SVM model.
    X : np.ndarray, shape (M, D)
        Test features.
        
    Returns
    -------
    y_pred : np.ndarray, shape (M,)
        Predicted labels {-1.0, +1.0}.
    """
    return model.predict(X)


# ==============================================================================
# 3. SAMPLE GENERATION
# ==============================================================================

def generate_samples(n_samples=60, dataset_type='linearly_separable', noise=0.2, random_state=42):
    """
    Generates 2D synthetic datasets for SVM classification.
    
    Per requirement: Defaults to 60 samples, with an options flag to choose the number.
    
    Parameters
    ----------
    n_samples : int, default=60
        Total number of sample points to generate.
    dataset_type : str, default='linearly_separable'
        Type of dataset:
          - 'linearly_separable' or 'linear': Two 2D Gaussian clusters (matching Example 1 in PDF).
          - 'xor': 4 alternating clusters in the 4 quadrants (matching Example 2 in PDF).
          - 'circular': Inner circle vs outer concentric ring (matching Example 3 in PDF).
    noise : float, default=0.2
        Standard deviation of Gaussian noise added to points.
    random_state : int or None, default=42
        Random seed for reproducible results.
        
    Returns
    -------
    X : np.ndarray, shape (n_samples, 2)
        Generated 2D feature coordinates.
    y : np.ndarray, shape (n_samples,)
        Class labels {-1.0, +1.0}.
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = int(n_samples)
    if n_samples < 2:
        raise ValueError("Number of samples must be at least 2.")

    dataset_norm = str(dataset_type).lower().replace('-', '_')

    if dataset_norm in ('linearly_separable', 'linear', 'blobs'):
        # Two Gaussian clusters matching Example 1 from the lecture notes (Page 6)
        n1 = n_samples // 2
        n2 = n_samples - n1

        # Class +1 points centered around (1.8, 2.0)
        cov = np.array([[0.8, 0.3], [0.3, 0.8]])
        X1 = np.random.multivariate_normal(mean=[1.8, 2.0], cov=cov, size=n1)
        y1 = np.ones(n1, dtype=np.float64)

        # Class -1 points centered around (-1.8, -2.0)
        X2 = np.random.multivariate_normal(mean=[-1.8, -2.0], cov=cov, size=n2)
        y2 = -np.ones(n2, dtype=np.float64)

        X = np.vstack([X1, X2])
        y = np.concatenate([y1, y2])

    elif dataset_norm == 'xor':
        # 4 clusters corresponding to the XOR distribution in Example 2 (Page 6-7)
        # Quadrants 1 & 3: Class +1; Quadrants 2 & 4: Class -1
        pts_per_cluster = n_samples // 4
        rem = n_samples % 4

        c1 = np.random.randn(pts_per_cluster + (1 if rem > 0 else 0), 2) * 0.55 + np.array([2.0, 2.0])   # Q1: +1
        c2 = np.random.randn(pts_per_cluster + (1 if rem > 1 else 0), 2) * 0.55 + np.array([-2.0, -2.0]) # Q3: +1
        c3 = np.random.randn(pts_per_cluster + (1 if rem > 2 else 0), 2) * 0.55 + np.array([-2.0, 2.0])  # Q2: -1
        c4 = np.random.randn(pts_per_cluster, 2) * 0.55 + np.array([2.0, -2.0])                          # Q4: -1

        X = np.vstack([c1, c2, c3, c4])
        y = np.concatenate([
            np.ones(len(c1) + len(c2), dtype=np.float64),
            -np.ones(len(c3) + len(c4), dtype=np.float64)
        ])

    elif dataset_norm in ('circular', 'circles'):
        # Concentric circles matching Example 3 (Page 7)
        n1 = n_samples // 2
        n2 = n_samples - n1

        # Inner circle: Class +1
        theta1 = np.random.uniform(0, 2 * np.pi, n1)
        r1 = np.random.uniform(0.1, 1.2, n1)
        X1 = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)])
        y1 = np.ones(n1, dtype=np.float64)

        # Outer ring: Class -1
        theta2 = np.random.uniform(0, 2 * np.pi, n2)
        r2 = np.random.uniform(2.0, 3.2, n2)
        X2 = np.column_stack([r2 * np.cos(theta2), r2 * np.sin(theta2)])
        y2 = -np.ones(n2, dtype=np.float64)

        X = np.vstack([X1, X2]) + np.random.randn(n_samples, 2) * (noise * 0.2)
        y = np.concatenate([y1, y2])

    else:
        raise ValueError(f"Unknown dataset_type '{dataset_type}'. Options: 'linearly_separable', 'xor', 'circular'.")

    # Shuffle the samples randomly
    perm = np.random.permutation(n_samples)
    return X[perm], y[perm]


# ==============================================================================
# 4. VISUALIZATION OF DATA POINTS, BOUNDARIES, AND MARGINS
# ==============================================================================

def visualize_svm_boundary(model, X, y, title=None, save_path=None, show=True):
    """
    Visualizes 2D data points, decision boundary, margins, and support vectors.
    
    Faithfully reproduces the visual format in Examples 1 to 4 of the lecture notes:
      - Class +1 and Class -1 points represented by distinct colored markers.
      - Support vectors highlighted as prominent solid markers.
      - Decision boundary f(x) = 0 plotted as a solid line.
      - Margins f(x) = +1 and f(x) = -1 plotted as dashed lines.
      - Decision regions shaded with soft color contours (blue vs red regions).
      
    Parameters
    ----------
    model : SMOSVM
        Fitted SVM model instance.
    X : np.ndarray, shape (N, 2)
        2D data points.
    y : np.ndarray, shape (N,)
        Class labels {-1, +1}.
    title : str or None, default=None
        Custom plot title. If None, automatically formatted with model parameters.
    save_path : str or None, default=None
        File path where the plot should be saved (e.g., 'svm_result.png').
    show : bool, default=True
        Whether to display the figure interactively via plt.show().
        
    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects
    """
    fig, ax = plt.subplots(figsize=(8, 7), dpi=120)

    # Calculate grid bounds with a 15% margin
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    dx = (x_max - x_min) * 0.15 or 1.0
    dy = (y_max - y_min) * 0.15 or 1.0
    x_min, x_max = x_min - dx, x_max + dx
    y_min, y_max = y_min - dy, y_max + dy

    # Create dense 2D mesh grid
    grid_res = 200
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_res),
                         np.linspace(y_min, y_max, grid_res))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Evaluate decision function over the grid
    Z = model.decision_function(grid_points)
    Z = Z.reshape(xx.shape)

    # 1. Fill background regions corresponding to class +1 and class -1 predictions
    # Matches the blue/red partitioning from Example 2 & 3 in the PDF
    ax.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9],
                colors=['#d6e4ff', '#ffd8d8'], alpha=0.6)

    # 2. Draw margin contours: f(x) = -1, f(x) = 0 (boundary), and f(x) = +1
    contour_lines = ax.contour(xx, yy, Z, levels=[-1.0, 0.0, 1.0],
                               colors=['#0052cc', '#111111', '#cc0000'],
                               linestyles=['--', '-', '--'],
                               linewidths=[1.8, 2.5, 1.8])
    
    # Label contour lines
    ax.clabel(contour_lines, inline=True, fontsize=10, fmt={-1.0: 'f(x) = -1', 0.0: 'f(x) = 0', 1.0: 'f(x) = +1'})

    # 3. Plot non-support vector training points
    sv_indices_set = set(model.support_indices)
    non_sv_mask = np.array([idx not in sv_indices_set for idx in range(len(y))])

    # Class +1 non-support vectors (red hollow circles)
    mask_pos_nonsv = non_sv_mask & (y > 0)
    ax.scatter(X[mask_pos_nonsv, 0], X[mask_pos_nonsv, 1],
               c='#d9381e', edgecolors='#900c3f', marker='o', s=55,
               alpha=0.75, label='Class +1 (Regular)')

    # Class -1 non-support vectors (blue hollow circles)
    mask_neg_nonsv = non_sv_mask & (y < 0)
    ax.scatter(X[mask_neg_nonsv, 0], X[mask_neg_nonsv, 1],
               c='#2b7de9', edgecolors='#104e8b', marker='o', s=55,
               alpha=0.75, label='Class -1 (Regular)')

    # 4. Highlight Support Vectors (Solid dots with a distinct ring, matching PDF Examples)
    if model.nsv > 0:
        sv_pos = (model.support_y > 0)
        sv_neg = (model.support_y < 0)

        # Class +1 support vectors
        if np.any(sv_pos):
            ax.scatter(model.support_X[sv_pos, 0], model.support_X[sv_pos, 1],
                       c='#b30000', edgecolors='black', linewidths=1.8,
                       marker='o', s=110, zorder=5, label=f'Class +1 SV ({np.sum(sv_pos)})')

        # Class -1 support vectors
        if np.any(sv_neg):
            ax.scatter(model.support_X[sv_neg, 0], model.support_X[sv_neg, 1],
                       c='#003399', edgecolors='black', linewidths=1.8,
                       marker='o', s=110, zorder=5, label=f'Class -1 SV ({np.sum(sv_neg)})')

    # Formatting and aesthetics
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Feature $x_1$', fontsize=12)
    ax.set_ylabel('Feature $x_2$', fontsize=12)

    # Compute training accuracy
    y_pred = model.predict(X)
    acc = np.mean(y_pred == y) * 100.0

    if title is None:
        kname = {'l': 'Linear', 'g': 'Gaussian', 'p': 'Polynomial', 'rbf_sq': 'Squared RBF'}.get(model.kernel, model.kernel)
        title = (f"SVM SMO Classification: {kname} Kernel\n"
                 f"$C = {model.C}$, Total Samples = {len(y)}, Support Vectors = {model.nsv} ({acc:.1f}% Accuracy)")
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)

    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"[SMO SVM] Visualization saved to: {save_path}")

    if show:
        plt.show()

    return fig, ax


# ==============================================================================
# 5. COMMAND-LINE INTERFACE (CLI)
# ==============================================================================

def parse_args():
    """Parses command-line options for running the SVM SMO algorithm."""
    parser = argparse.ArgumentParser(
        description="Sequential Minimal Optimization (SMO) algorithm for SVM binary classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Options flag to choose number of samples (per default 60)
    parser.add_argument(
        '-n', '--samples', '--num_samples',
        dest='n_samples',
        type=int,
        default=60,
        help="Number of samples to generate (default: 60)."
    )
    # Regularization parameter C option
    parser.add_argument(
        '-c', '--C',
        dest='C',
        type=float,
        default=1.2,
        help="Soft-margin regularization parameter C."
    )
    # Kernel selection
    parser.add_argument(
        '-k', '--kernel',
        dest='kernel',
        type=str,
        default='l',
        choices=['l', 'linear', 'g', 'gaussian', 'p', 'polynomial', 'rbf_sq'],
        help="Kernel function: 'l' (linear), 'g' (gaussian), 'p' (polynomial), 'rbf_sq' (squared RBF)."
    )
    # Gaussian kernel parameter gamma (MATLAB 'gm')
    parser.add_argument(
        '-g', '--gamma',
        dest='gamma',
        type=float,
        default=0.5,
        help="Gaussian / RBF kernel bandwidth parameter gamma."
    )
    # Polynomial kernel degree
    parser.add_argument(
        '--degree',
        type=int,
        default=3,
        help="Degree for polynomial kernel."
    )
    # Dataset type selection
    parser.add_argument(
        '-d', '--dataset',
        dest='dataset',
        type=str,
        default='linearly_separable',
        choices=['linearly_separable', 'linear', 'xor', 'circular'],
        help="Synthetic dataset distribution."
    )
    # Numerical tolerance
    parser.add_argument(
        '--tol',
        type=float,
        default=1e-3,
        help="Convergence and KKT violation tolerance."
    )
    # Maximum iterations
    parser.add_argument(
        '--max_iter',
        type=int,
        default=100,
        help="Maximum number of passes over dataset."
    )
    # Random seed
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for reproducible sample generation."
    )
    # Output file to save visualization
    parser.add_argument(
        '--save_fig',
        type=str,
        default='svm_smo_boundary.png',
        help="File path to save the generated boundary visualization plot."
    )
    # Headless / no interactive window display
    parser.add_argument(
        '--no_plot',
        action='store_true',
        help="Disable interactive graphical display window (useful in headless environments)."
    )
    return parser.parse_args()


def main():
    """CLI execution entrypoint."""
    args = parse_args()

    print("=================================================================")
    print(" Sequential Minimal Optimization (SMO) SVM Classifier")
    print("=================================================================")
    print(f"Generating samples      : {args.n_samples} samples (dataset: '{args.dataset}', seed: {args.seed})")
    print(f"Regularization parameter: C = {args.C}")
    print(f"Kernel function         : '{args.kernel}' (gamma = {args.gamma}, degree = {args.degree})")
    print(f"Tolerance / Max Iter    : tol = {args.tol}, max_iter = {args.max_iter}")
    print("-----------------------------------------------------------------")

    # 1. Generate synthetic samples
    X, y = generate_samples(
        n_samples=args.n_samples,
        dataset_type=args.dataset,
        random_state=args.seed
    )
    print(f"Dataset generated: {len(y)} samples (Class +1: {np.sum(y == 1)}, Class -1: {np.sum(y == -1)})")

    # 2. Train the SMO SVM model
    svm = SMOSVM(
        C=args.C,
        kernel=args.kernel,
        gamma=args.gamma,
        degree=args.degree,
        tol=args.tol,
        max_iter=args.max_iter
    )
    print("Training SVM using SMO algorithm...")
    svm.fit(X, y)

    # 3. Output results
    y_pred = svm.predict(X)
    accuracy = np.mean(y_pred == y) * 100.0
    print("-----------------------------------------------------------------")
    print(f"Optimization complete!")
    print(f"Support Vectors count   : {svm.nsv} / {len(y)} ({svm.nsv / len(y) * 100:.1f}%)")
    print(f"Model bias (offset b)   : {svm.bias:.5f}")
    print(f"Training accuracy       : {accuracy:.2f}%")
    print("=================================================================")

    # 4. Visualize data points, boundaries, and margins
    visualize_svm_boundary(
        model=svm,
        X=X,
        y=y,
        save_path=args.save_fig,
        show=not args.no_plot
    )


if __name__ == '__main__':
    main()
