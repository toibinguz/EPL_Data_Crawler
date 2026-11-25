"""Preprocessing: SimpleImputer(median) -> ClipPercentile(1-99) -> StandardScaler."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ClipPercentile(BaseEstimator, TransformerMixin):
    """Clip every column to [lower, upper] percentiles learned on the fit set (train)."""

    def __init__(self, lower=1.0, upper=99.0):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanpercentile(X, self.lower, axis=0)
        self.hi_ = np.nanpercentile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def make_prep():
    return [("imp", SimpleImputer(strategy="median")),
            ("clip", ClipPercentile(1, 99)),
            ("sc", StandardScaler())]


def lr_pipe(C):
    return Pipeline(make_prep() + [("lr", LogisticRegression(C=C, max_iter=5000))])
