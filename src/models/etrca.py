"""Ensemble TRCA (eTRCA) classifier for SSVEP."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class ETRCA(BaseEstimator, ClassifierMixin):
    """Ensemble Task-Related Component Analysis classifier.

    Parameters
    ----------
    n_components : int
        Number of spatial filters per class.
    """

    def __init__(self, n_components: int = 1):
        self.n_components = n_components
        self.spatial_filters_ = {}
        self.templates_ = {}
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit spatial filters and templates per class.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples)
        y : (n_trials,)
        """
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            trials = X[y == cls]          # (n_cls_trials, n_channels, n_samples)
            self.templates_[cls] = trials.mean(axis=0)
            W = self._compute_trca(trials)
            self.spatial_filters_[cls] = W
        return self

    def _compute_trca(self, trials: np.ndarray) -> np.ndarray:
        n_trials, n_ch, n_s = trials.shape
        S = np.zeros((n_ch, n_ch))
        for i in range(n_trials):
            for j in range(i + 1, n_trials):
                S += trials[i] @ trials[j].T
        Q = sum(trials[i] @ trials[i].T for i in range(n_trials))
        eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(Q) @ S)
        idx = np.argsort(eigvals)[::-1]
        return eigvecs[:, idx[:self.n_components]]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : (n_trials, n_channels, n_samples)
        """
        preds = []
        for trial in X:
            scores = {}
            for cls in self.classes_:
                W = self.spatial_filters_[cls]
                template = self.templates_[cls]
                r = np.corrcoef(
                    (W.T @ trial).ravel(),
                    (W.T @ template).ravel()
                )[0, 1]
                scores[cls] = r
            preds.append(max(scores, key=scores.get))
        return np.array(preds)
