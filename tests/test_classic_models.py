#!/usr/bin/env python
# Created by "Thieu" at 00:56, 17/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from pylwl import LwClassifier, LwRegressor


@pytest.fixture
def binary_classification_data():
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def multiclass_classification_data():
    X, y = make_classification(n_samples=120, n_features=10, n_classes=3, n_informative=4, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_lw_classifier_binary(binary_classification_data):
    X_train, X_test, y_train, y_test = binary_classification_data
    clf = LwClassifier(kernel="gaussian", tau=0.5)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    probas = clf.predict_proba(X_test)

    assert y_pred.shape == y_test.shape
    assert probas.shape == (len(y_test), 2)
    assert np.allclose(probas.sum(axis=1), 1, atol=1e-5)

    score = clf.score(X_test, y_test)
    assert 0 <= score <= 1

    metrics = clf.scores(X_test, y_test, list_metrics=["AS"])
    assert "AS" in metrics
    assert 0 <= metrics["AS"] <= 1


def test_lw_classifier_multiclass(multiclass_classification_data):
    X_train, X_test, y_train, y_test = multiclass_classification_data
    clf = LwClassifier(kernel="gaussian", tau=1.0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    probas = clf.predict_proba(X_test)

    assert y_pred.shape == y_test.shape
    assert probas.shape == (len(y_test), len(np.unique(y_train)))
    assert np.allclose(probas.sum(axis=1), 1, atol=1e-5)

    score = clf.score(X_test, y_test)
    assert 0 <= score <= 1

    metrics = clf.scores(X_test, y_test, list_metrics=["AS"])
    assert "AS" in metrics
    assert 0 <= metrics["AS"] <= 1


def test_lw_regressor_basic(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    reg = LwRegressor(kernel="gaussian", tau=0.3)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    assert y_pred.shape == y_test.shape
    assert np.isfinite(y_pred).all()

    score = reg.score(X_test, y_test)
    assert -1 <= score <= 1

    metrics = reg.scores(X_test, y_test, list_metrics=["MSE", "MAE"])
    assert "MSE" in metrics
    assert "MAE" in metrics
    assert metrics["MSE"] >= 0
    assert metrics["MAE"] >= 0
