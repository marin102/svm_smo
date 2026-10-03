#!/usr/bin/env python3
"""
Demonstration: Replicating Example 1 from Prof. Ruye Wang's Lecture Notes
=========================================================================
Shows the comparison between:
- Large C = 1.2  (narrow margin, fewer support vectors)
- Small C = 0.05 (wide margin, more support vectors)
on the exact same dataset of 60 samples.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from svm_smo import SMOSVM, generate_samples


def run_example1_comparison(save_path='svm_example1_comparison.png'):
    # Generate identical 60 samples for both models
    X, y = generate_samples(n_samples=60, dataset_type='linearly_separable', random_state=42)

    # Train SVM with large C (1.2)
    svm_large = SMOSVM(C=1.2, kernel='l', max_iter=100).fit(X, y)

    # Train SVM with small C (0.05)
    svm_small = SMOSVM(C=0.05, kernel='l', max_iter=100).fit(X, y)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=130)

    # Grid bounds
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    for ax, svm, c_val in zip(axes, [svm_large, svm_small], [1.2, 0.05]):
        Z = svm.decision_function(grid).reshape(xx.shape)

        # Decision regions
        ax.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9], colors=['#d6e4ff', '#ffd8d8'], alpha=0.55)

        # Boundary f(x)=0 and margins f(x)=+-1
        cl = ax.contour(xx, yy, Z, levels=[-1.0, 0.0, 1.0],
                        colors=['#0052cc', '#111111', '#cc0000'],
                        linestyles=['--', '-', '--'], linewidths=[1.8, 2.5, 1.8])
        ax.clabel(cl, inline=True, fontsize=9, fmt={-1.0: 'f(x)=-1', 0.0: 'f(x)=0', 1.0: 'f(x)=+1'})

        # Regular points
        sv_set = set(svm.support_indices)
        nonsv_pos = [i for i in range(len(y)) if y[i] > 0 and i not in sv_set]
        nonsv_neg = [i for i in range(len(y)) if y[i] < 0 and i not in sv_set]

        ax.scatter(X[nonsv_pos, 0], X[nonsv_pos, 1], c='#d9381e', marker='o', s=50, alpha=0.75, label='Class +1')
        ax.scatter(X[nonsv_neg, 0], X[nonsv_neg, 1], c='#2b7de9', marker='o', s=50, alpha=0.75, label='Class -1')

        # Support vectors (solid dots with black ring)
        if svm.nsv > 0:
            sv_pos = [i for i in svm.support_indices if y[i] > 0]
            sv_neg = [i for i in svm.support_indices if y[i] < 0]
            if sv_pos:
                ax.scatter(X[sv_pos, 0], X[sv_pos, 1], c='#b30000', edgecolors='black', linewidths=1.8,
                           marker='o', s=110, zorder=5, label=f'Class +1 SV ({len(sv_pos)})')
            if sv_neg:
                ax.scatter(X[sv_neg, 0], X[sv_neg, 1], c='#003399', edgecolors='black', linewidths=1.8,
                           marker='o', s=110, zorder=5, label=f'Class -1 SV ({len(sv_neg)})')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('$x_1$', fontsize=11)
        ax.set_ylabel('$x_2$', fontsize=11)
        ax.set_title(f"Example 1: C = {c_val} (Support Vectors: {svm.nsv})", fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8.5)
        ax.grid(True, linestyle=':', alpha=0.45)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    print(f"Saved side-by-side comparison to {save_path}")


if __name__ == '__main__':
    run_example1_comparison()
