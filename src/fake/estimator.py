import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data


class FakeEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def fit(self, X, y):
        # Check that X and y have correct shape, set n_features_in_, etc.
        X, y = validate_data(self, X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        probs = np.random.rand(len(X), len(self.classes_))
        normalized_probs = probs / np.sum(probs, axis=1, keepdims=True)
        return normalized_probs

    def predict(self, X):
        probs = self.predict_proba(X)
        predictions = np.argmax(probs, axis=1)
        return predictions
