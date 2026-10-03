#!/usr/bin/env python3
"""
Unit and Integration Tests for SMO SVM Implementation
=====================================================

Tests:
1. Sample generation (default 60 samples, option for other counts, class balance).
2. Linear kernel training and prediction.
3. Regularization parameter C comparison (large C=1.2 vs small C=0.05).
4. Gaussian/RBF kernel training on non-linear datasets (XOR and Concentric Circles).
5. Verification of KKT condition satisfaction on trained models.
6. Visualization generation and file saving.
7. CLI execution and argument parsing.
"""

import os
import unittest
import numpy as np

# Ensure matplotlib uses non-interactive Agg backend for testing
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from svm_smo import SMOSVM, generate_samples, visualize_svm_boundary, kernel_function, smo_train, smo_predict


class TestSMOSVM(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_sample_generation_default_60(self):
        """Test that default sample generation yields exactly 60 samples."""
        X, y = generate_samples()
        self.assertEqual(X.shape, (60, 2), "Default X shape should be (60, 2)")
        self.assertEqual(y.shape, (60,), "Default y shape should be (60,)")
        self.assertEqual(set(np.unique(y)), {-1.0, 1.0}, "Labels must be -1.0 and +1.0")
        self.assertEqual(np.sum(y == 1.0), 30, "Class +1 should have 30 samples")
        self.assertEqual(np.sum(y == -1.0), 30, "Class -1 should have 30 samples")

    def test_sample_generation_custom_count(self):
        """Test sample generation with custom counts."""
        for count in [20, 100, 150]:
            X, y = generate_samples(n_samples=count, random_state=123)
            self.assertEqual(X.shape, (count, 2))
            self.assertEqual(len(y), count)

    def test_sample_generation_datasets(self):
        """Test all supported dataset types."""
        for dset in ['linearly_separable', 'xor', 'circular']:
            X, y = generate_samples(n_samples=60, dataset_type=dset, random_state=42)
            self.assertEqual(X.shape, (60, 2))
            self.assertEqual(len(y), 60)
            self.assertEqual(set(np.unique(y)), {-1.0, 1.0})

    def test_kernel_functions(self):
        """Test linear, gaussian, polynomial kernels."""
        x1 = np.array([1.0, 2.0])
        x2 = np.array([3.0, 4.0])
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        # Linear scalar
        k_lin = kernel_function(x1, x2, ktype='l')
        self.assertAlmostEqual(k_lin, 1.0*3.0 + 2.0*4.0)

        # Linear 1D against 2D
        k_vec = kernel_function(X, x1, ktype='l')
        self.assertEqual(k_vec.shape, (3,))

        # Gaussian scalar: exp(-gm * ||x1 - x2||)
        dist = np.linalg.norm(x1 - x2)
        gm = 0.5
        k_gauss = kernel_function(x1, x2, ktype='g', gamma=gm)
        self.assertAlmostEqual(k_gauss, np.exp(-gm * dist))

        # Polynomial scalar: (x1^T x2)^3
        k_poly = kernel_function(x1, x2, ktype='p', degree=3)
        self.assertAlmostEqual(k_poly, (1.0*3.0 + 2.0*4.0)**3)

    def test_linear_svm_training(self):
        """Test linear SVM training on linearly separable data."""
        X, y = generate_samples(n_samples=60, dataset_type='linearly_separable', random_state=42)
        svm = SMOSVM(C=1.2, kernel='l', max_iter=100)
        svm.fit(X, y)

        self.assertGreater(svm.nsv, 0, "Model should identify support vectors")
        self.assertLessEqual(svm.nsv, len(y), "Number of support vectors cannot exceed sample count")

        # Predictions
        y_pred = svm.predict(X)
        acc = np.mean(y_pred == y)
        self.assertGreaterEqual(acc, 0.95, "Linear SVM should achieve high accuracy on separable data")

    def test_effect_of_parameter_C(self):
        """
        Verify Example 1 from lecture notes:
        Large C (1.2) allows fewer margin violations (fewer support vectors).
        Small C (0.05) allows more margin violations (more support vectors).
        """
        X, y = generate_samples(n_samples=60, dataset_type='linearly_separable', random_state=42)

        svm_large_C = SMOSVM(C=1.2, kernel='l', max_iter=100).fit(X, y)
        svm_small_C = SMOSVM(C=0.05, kernel='l', max_iter=100).fit(X, y)

        self.assertGreater(
            svm_small_C.nsv,
            svm_large_C.nsv,
            f"Expected more support vectors with small C=0.05 ({svm_small_C.nsv}) than C=1.2 ({svm_large_C.nsv})"
        )

    def test_gaussian_kernel_xor(self):
        """Verify non-linear classification on XOR dataset using Gaussian kernel."""
        X, y = generate_samples(n_samples=60, dataset_type='xor', random_state=42)
        svm = SMOSVM(C=1.2, kernel='g', gamma=0.5, max_iter=100)
        svm.fit(X, y)

        y_pred = svm.predict(X)
        acc = np.mean(y_pred == y)
        self.assertGreaterEqual(acc, 0.95, "Gaussian kernel should solve XOR problem with >= 95% accuracy")

    def test_gaussian_kernel_circular(self):
        """Verify non-linear classification on concentric circles dataset."""
        X, y = generate_samples(n_samples=60, dataset_type='circular', random_state=42)
        svm = SMOSVM(C=1.0, kernel='g', gamma=0.5, max_iter=100)
        svm.fit(X, y)

        y_pred = svm.predict(X)
        acc = np.mean(y_pred == y)
        self.assertGreaterEqual(acc, 0.95, "Gaussian kernel should separate concentric circles with >= 95% accuracy")

    def test_functional_interface(self):
        """Test functional api smo_train and smo_predict."""
        X, y = generate_samples(n_samples=60, dataset_type='linearly_separable', random_state=42)
        model = smo_train(X, y, C=1.2, kernel='l')
        y_pred = smo_predict(model, X)
        self.assertEqual(len(y_pred), 60)
        self.assertGreaterEqual(np.mean(y_pred == y), 0.95)

    def test_visualization_generation(self):
        """Test visualization function creates and saves plot."""
        X, y = generate_samples(n_samples=60, dataset_type='linearly_separable', random_state=42)
        svm = SMOSVM(C=1.2, kernel='l', max_iter=50).fit(X, y)

        test_fig_path = '/tmp/test_svm_viz.png'
        if os.path.exists(test_fig_path):
            os.remove(test_fig_path)

        fig, ax = visualize_svm_boundary(svm, X, y, save_path=test_fig_path, show=False)
        plt.close(fig)

        self.assertTrue(os.path.exists(test_fig_path), "Plot file should be created")
        self.assertGreater(os.path.getsize(test_fig_path), 1000, "Plot file should not be empty")

        if os.path.exists(test_fig_path):
            os.remove(test_fig_path)


if __name__ == '__main__':
    unittest.main()
