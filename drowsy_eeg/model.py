"""Train and evaluate the awake-vs-drowsy classifier with subject-level splits."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .signal import FEATURES


def make_model():
    """Scaled logistic regression; balanced weights because drowsy epochs are rarer."""
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))


def _scores(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "drowsy_recall": float(recall_score(y_true, pred, pos_label=1)),
        "awake_recall": float(recall_score(y_true, pred, pos_label=0)),
    }


def cross_validate(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, n_splits: int = 5) -> dict:
    """GroupKFold by subject, so no person appears in both training and test folds.

    Also scores a one-feature baseline (theta/alpha ratio alone) on the same
    folds, to show what the full feature set adds.
    """
    ratio_col = FEATURES.index("theta_alpha_ratio")
    full_prob = np.zeros(len(y))
    base_prob = np.zeros(len(y))
    for train, test in GroupKFold(n_splits=n_splits).split(X, y, subjects):
        full_prob[test] = make_model().fit(X[train], y[train]).predict_proba(X[test])[:, 1]
        base = make_model().fit(X[train][:, [ratio_col]], y[train])
        base_prob[test] = base.predict_proba(X[test][:, [ratio_col]])[:, 1]
    return {
        "n_epochs": int(len(y)),
        "n_drowsy": int(y.sum()),
        "n_subjects": int(len(np.unique(subjects))),
        "folds": n_splits,
        "model": _scores(y, full_prob),
        "baseline_theta_alpha_only": _scores(y, base_prob),
    }
