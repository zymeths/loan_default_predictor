"""
evaluate.py
-----------
Evaluation utilities for the loan default predictor.

Usage:
    from src.evaluate import plot_feature_importance, plot_oof_distribution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from lightgbm import LGBMClassifier


def plot_feature_importance(model: LGBMClassifier,
                             feature_names: list,
                             top_n: int = 20) -> None:
    """Bar chart of top N feature importances from a fitted LGBMClassifier."""
    imp = pd.Series(model.feature_importances_, index=feature_names)
    imp = imp.sort_values().tail(top_n)

    fig, ax = plt.subplots(figsize=(8, top_n * 0.35))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    imp.plot(kind='barh', ax=ax, color='#58a6ff')
    ax.set_title(f'Top {top_n} Feature Importances', color='#c9d1d9', pad=12)
    ax.tick_params(colors='#8b949e')
    ax.set_xlabel('Importance', color='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Plot ROC curve with AUC score."""
    auc  = roc_auc_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    ax.plot(fpr, tpr, color='#58a6ff', lw=2, label=f'AUC = {auc:.4f}')
    ax.plot([0, 1], [0, 1], color='#30363d', lw=1, linestyle='--')
    ax.set_xlabel('False Positive Rate', color='#8b949e')
    ax.set_ylabel('True Positive Rate', color='#8b949e')
    ax.set_title('ROC Curve (OOF)', color='#c9d1d9', pad=12)
    ax.tick_params(colors='#8b949e')
    ax.legend(facecolor='#161b22', labelcolor='#c9d1d9')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

    plt.tight_layout()
    plt.show()


def plot_oof_distribution(y_true: np.ndarray, oof_preds: np.ndarray) -> None:
    """Histogram of OOF predicted probabilities split by actual class."""
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    ax.hist(oof_preds[y_true == 1], bins=60, alpha=0.6,
            color='#58a6ff', label='Paid Back (1)')
    ax.hist(oof_preds[y_true == 0], bins=60, alpha=0.6,
            color='#f85149', label='Default (0)')
    ax.set_xlabel('Predicted Probability', color='#8b949e')
    ax.set_ylabel('Count', color='#8b949e')
    ax.set_title('OOF Prediction Distribution by Class', color='#c9d1d9', pad=12)
    ax.tick_params(colors='#8b949e')
    ax.legend(facecolor='#161b22', labelcolor='#c9d1d9')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

    plt.tight_layout()
    plt.show()
