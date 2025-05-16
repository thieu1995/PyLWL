#!/usr/bin/env python
# Created by "Thieu" at 13:12, 16/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from scipy.special import expit, softmax  # for sigmoid
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from pylwl.models.base_model import BaseModel
from pylwl.shared import kernel as kernel_module


class BaseLW(BaseModel):
    """
    Base class for locally weighted models.

    This class provides the foundation for locally weighted regression and classification models.
    It includes methods for computing kernel weights based on a specified kernel function.

    Parameters
    ----------
    kernel : str or callable, optional
        The kernel function to use. If a string is provided, it should match the name of a kernel
        function in the `kernel_module`. If a callable is provided, it should accept distances
        and `tau` as arguments and return weights.
    tau : float, optional
        The bandwidth parameter for the kernel function (default: 1.0).

    Attributes
    ----------
    kernel : str or callable
        The kernel function used for computing weights.
    kernel_func_ : callable
        The resolved kernel function (either from `kernel_module` or the provided callable).
    tau : float
        The bandwidth parameter for the kernel function.
    """

    def __init__(self, kernel='gaussian', tau=1.0):
        super().__init__()
        self.kernel = kernel
        if isinstance(kernel, str):
            self.kernel_func_ = getattr(kernel_module, f"{kernel}_kernel")
        elif callable(kernel):
            self.kernel_func_ = kernel
        else:
            raise ValueError("kernel must be a string or callable")
        self.tau = tau

    def _kernel_weights(self, X_train, x_query):
        """
        Compute kernel weights for a query point.

        Parameters
        ----------
        X_train : array-like, shape (n_samples, n_features)
            The training data.
        x_query : array-like, shape (n_features,)
            The query point.

        Returns
        -------
        W : ndarray, shape (n_samples, n_samples)
            A diagonal matrix of kernel weights for the query point.
        """
        distances = np.linalg.norm(X_train - x_query, axis=1)
        weights = self.kernel_func_(distances, tau=self.tau)
        W = np.diag(weights)
        W = np.clip(W, 1e-8, 1e8)
        return W


class LwRegressor(BaseLW, RegressorMixin):
    """
    Locally Weighted Regressor.

    This class implements a locally weighted regression model using a specified kernel function
    and bandwidth parameter. It predicts target values by fitting a weighted linear model
    for each query point.

    Parameters
    ----------
    kernel : str or callable, optional
        The kernel function to use. If a string is provided, it should match the name of a kernel
        function in the `kernel_module`. If a callable is provided, it should accept distances
        and `tau` as arguments and return weights.
    tau : float, optional
        The bandwidth parameter for the kernel function (default: 1.0).

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The training data.
    y_ : ndarray, shape (n_samples,)
        The target values for the training data.
    """

    def __init__(self, kernel="gaussian", tau=1.0):
        """
        Initialize the LwRegressor.

        Parameters
        ----------
        kernel : str or callable, optional
            The kernel function to use (default: "gaussian").
        tau : float, optional
            The bandwidth parameter for the kernel function (default: 1.0).
        """
        super().__init__(kernel=kernel, tau=tau)

    def fit(self, X, y):
        """
        Fit the locally weighted regression model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training data.
        y : array-like, shape (n_samples,)
            The target values.

        Returns
        -------
        self : LwRegressor
            The fitted model.
        """
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """
        Predict target values for the given input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            The predicted target values.
        """
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        y_preds = []
        for x in X:
            W = self._kernel_weights(self.X_, x)
            X_aug = np.hstack([np.ones((self.X_.shape[0], 1)), self.X_])
            x_aug = np.insert(x, 0, 1)
            try:
                theta = np.linalg.pinv(X_aug.T @ W @ X_aug) @ X_aug.T @ W @ self.y_
                y_pred = x_aug @ theta
            except np.linalg.LinAlgError:
                y_pred = np.mean(self.y_)
            y_preds.append(y_pred)
        return np.array(y_preds)

    def score(self, X, y):
        """
        Compute the R^2 score for the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        y : array-like, shape (n_samples,)
            The true target values.

        Returns
        -------
        score : float
            The R^2 score of the predictions.
        """
        return r2_score(y, self.predict(X))

    def evaluate(self, y_true, y_pred, list_metrics=("MSE", "MAE")):
        """
        Evaluate the regression model using specified metrics.

        Parameters
        ----------
        y_true : array-like
            True target values.
        y_pred : array-like
            Predicted target values.
        list_metrics : tuple of str, optional
            List of metrics for evaluation (default: ("MSE", "MAE")).

        Returns
        -------
        dict
            Dictionary of calculated metric values.
        """
        return self._evaluate_reg(y_true, y_pred, list_metrics)  # Call the evaluation method

    def scores(self, X, y, list_metrics=("MSE", "MAE")):
        """
        Compute evaluation metrics for the model on the given data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        y : array-like, shape (n_samples,)
            The true target values.
        list_metrics : tuple of str, optional
            List of metrics for evaluation (default: ("MSE", "MAE")).

        Returns
        -------
        dict
            Dictionary of calculated metric values.
        """
        y_pred = self.predict(X)
        return self.evaluate(y, y_pred, list_metrics)


class LwClassifier(BaseLW, ClassifierMixin):
    """
    Locally Weighted Classifier.

    This class implements a locally weighted classification model using a specified kernel function
    and bandwidth parameter. It predicts class probabilities and labels by fitting a weighted linear
    model for each query point.

    Parameters
    ----------
    kernel : str or callable, optional
        The kernel function to use. If a string is provided, it should match the name of a kernel
        function in the `kernel_module`. If a callable is provided, it should accept distances
        and `tau` as arguments and return weights.
    tau : float, optional
        The bandwidth parameter for the kernel function (default: 1.0).

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The training data.
    y_raw_ : ndarray, shape (n_samples,)
        The raw target values for the training data.
    classes_ : ndarray, shape (n_classes,)
        The unique class labels.
    n_classes_ : int
        The number of unique classes.
    lb_ : LabelBinarizer
        The label binarizer used for encoding class labels.
    y_bin_ : ndarray, shape (n_samples, n_classes) or (n_samples,)
        The binarized target values for the training data.
    get_prob : callable
        The method used to compute class probabilities (binary or multiclass).
    """

    def __init__(self, kernel="gaussian", tau=1.0):
        """
        Initialize the LwClassifier.

        Parameters
        ----------
        kernel : str or callable, optional
            The kernel function to use (default: "gaussian").
        tau : float, optional
            The bandwidth parameter for the kernel function (default: 1.0).
        """
        super().__init__(kernel=kernel, tau=tau)

    def fit(self, X, y):
        """
        Fit the locally weighted classification model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training data.
        y : array-like, shape (n_samples,)
            The target class labels.

        Returns
        -------
        self : LwClassifier
            The fitted model.
        """
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_raw_ = y
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        self.lb_ = LabelBinarizer()
        self.y_bin_ = self.lb_.fit_transform(y)
        if self.n_classes_ == 2:
            self.y_bin_ = self.y_bin_.ravel()
            self.get_prob = self._get_binary
        else:
            self.get_prob = self._get_multiclass
        return self

    def _get_binary(self, logits):
        """
        Compute binary class probabilities.

        Parameters
        ----------
        logits : list of float
            The logits for the binary classification.

        Returns
        -------
        list
            The probabilities for each class.
        """
        prob = expit(logits[0])
        return [1 - prob, prob]

    def _get_multiclass(self, logits):
        """
        Compute multiclass probabilities.

        Parameters
        ----------
        logits : list of float
            The logits for the multiclass classification.

        Returns
        -------
        ndarray
            The probabilities for each class.
        """
        probs = softmax(logits)
        return probs

    def predict_proba(self, X):
        """
        Predict class probabilities for the given input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        probas : ndarray, shape (n_samples, n_classes)
            The predicted class probabilities.
        """
        check_is_fitted(self, ['X_', 'y_bin_'])
        X = check_array(X)
        X_aug = np.hstack([np.ones((self.X_.shape[0], 1)), self.X_])
        probas = []
        for x in X:
            W = self._kernel_weights(self.X_, x)
            x_aug = np.insert(x, 0, 1)
            logits = []
            for k in range(self.n_classes_):
                y_k = self.y_bin_[:, k] if self.n_classes_ > 2 else self.y_bin_
                try:
                    theta = np.linalg.pinv(X_aug.T @ W @ X_aug) @ X_aug.T @ W @ y_k
                    logit = x_aug @ theta
                except np.linalg.LinAlgError:
                    logit = np.log(np.mean(y_k) / (1 - np.mean(y_k) + 1e-8))
                logits.append(logit)
            probas.append(self.get_prob(logits))
        return np.array(probas)

    def predict(self, X):
        """
        Predict class labels for the given input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            The predicted class labels.
        """
        probas = self.predict_proba(X)
        class_indices = np.argmax(probas, axis=1)
        return self.classes_[class_indices]

    def score(self, X, y):
        """
        Compute the accuracy score for the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        y : array-like, shape (n_samples,)
            The true target class labels.

        Returns
        -------
        score : float
            The accuracy score of the predictions.
        """
        return accuracy_score(y, self.predict(X))

    def evaluate(self, y_true, y_pred, list_metrics=("AS", "RS")):
        """
        Evaluate the classification model using specified metrics.

        Parameters
        ----------
        y_true : array-like
            True target class labels.
        y_pred : array-like
            Predicted class labels.
        list_metrics : tuple of str, optional
            List of metrics for evaluation (default: ("AS", "RS")).

        Returns
        -------
        dict
            Dictionary of calculated metric values.
        """
        return self._evaluate_cls(y_true=y_true, y_pred=y_pred, list_metrics=list_metrics)

    def scores(self, X, y, list_metrics=("AS", "RS")):
        """
        Compute evaluation metrics for the model on the given data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        y : array-like, shape (n_samples,)
            The true target class labels.
        list_metrics : tuple of str, optional
            List of metrics for evaluation (default: ("AS", "RS")).

        Returns
        -------
        dict
            Dictionary of calculated metric values.
        """
        y_pred = self.predict(X)
        return self.evaluate(y, y_pred, list_metrics)
