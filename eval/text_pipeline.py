"""Adapter that presents a bare (model, vectorizer) pickle pair as a single
sklearn-style estimator.

The shipped artifacts are `fake_news_model.pkl` (a LogisticRegression) and
`fake_news_vectorizer.pkl` (a TfidfVectorizer) stored separately. The eval
harness wants an object with ``classes_``, ``predict`` and ``predict_proba``.
"""

from __future__ import annotations

import numpy as np


class PipelineModel:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer

    @property
    def classes_(self):
        return list(self.model.classes_)

    def _transform(self, texts):
        return self.vectorizer.transform(texts)

    def predict(self, texts):
        return self.model.predict(self._transform(texts))

    def predict_proba(self, texts):
        return self.model.predict_proba(self._transform(texts))

    def fit(self, texts, labels):  # pragma: no cover - parity with sklearn
        X = self.vectorizer.fit_transform(texts)
        y = np.asarray(labels)
        self.model.fit(X, y)
        return self
