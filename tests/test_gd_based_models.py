#!/usr/bin/env python
# Created by "Thieu" at 01:00, 17/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from pylwl import GdLwClassifier, GdLwRegressor


@pytest.fixture
def cls_data():
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def reg_data():
    X, y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.mark.parametrize("kernel", ['gaussian'])
def test_gdlw_classifier_fit_predict_score(kernel, cls_data):
    X_train, X_test, y_train, y_test = cls_data
    clf = GdLwClassifier(kernel=kernel, tau=1.0, epochs=5, seed=0, verbose=False)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    assert preds.shape == (len(y_test),)
    assert set(preds).issubset(set(y_train))

    score = clf.score(X_test, y_test)
    assert 0.0 <= score <= 1.0

    scores = clf.scores(X_test, y_test, list_metrics=["AS", "RS"])
    assert "AS" in scores and "RS" in scores
    assert all(0.0 <= v <= 1.0 for v in scores.values())


@pytest.mark.parametrize("kernel", ['gaussian'])
def test_gdlw_regressor_fit_predict_score(kernel, reg_data):
    X_train, X_test, y_train, y_test = reg_data
    reg = GdLwRegressor(kernel=kernel, tau=1.0, epochs=5, seed=0, verbose=False)
    reg.fit(X_train, y_train)

    preds = reg.predict(X_test)
    assert preds.shape == (len(y_test),)
    assert np.isfinite(preds).all()

    score = reg.score(X_test, y_test)
    assert -1.0 <= score <= 1.0

    scores = reg.scores(X_test, y_test, list_metrics=["MSE", "MAE"])
    assert "MSE" in scores and "MAE" in scores
    assert all(v >= 0 for v in scores.values())


def test_classifier_predict_proba_shape(cls_data):
    X_train, X_test, y_train, y_test = cls_data
    clf = GdLwClassifier(tau=0.5, epochs=5, seed=0, verbose=False)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    assert proba.shape == (len(y_test), 2)
    assert np.allclose(proba.sum(axis=1), 1, atol=1e-4)


def test_regressor_prediction_stability(reg_data):
    X_train, X_test, y_train, y_test = reg_data
    reg = GdLwRegressor(tau=1.0, epochs=5, seed=123, verbose=False)
    reg.fit(X_train, y_train)
    preds1 = reg.predict(X_test)
    reg2 = GdLwRegressor(tau=1.0, epochs=5, seed=123, verbose=False)
    reg2.fit(X_train, y_train)
    preds2 = reg2.predict(X_test)
    np.testing.assert_allclose(preds1, preds2, rtol=1e-4)
