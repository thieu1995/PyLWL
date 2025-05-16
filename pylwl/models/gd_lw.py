#!/usr/bin/env python
# Created by "Thieu" at 15:17, 16/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from pylwl.shared import kernel as kernel_module
from pylwl.models.base_model import BaseModel


class GdLwClassifier(BaseModel, ClassifierMixin):
    """
    Gradient-Descent-based Locally Weighted Classifier.

    This class implements a locally weighted classification model using gradient descent
    optimization. It supports binary and multiclass classification with a specified kernel
    function and bandwidth parameter.

    Parameters
    ----------
    kernel : str or callable, optional
        The kernel function to use. If a string is provided, it should match the name of a kernel
        function in the `kernel_module`. If a callable is provided, it should accept distances
        and `tau` as arguments and return weights.
    tau : float, optional
        The bandwidth parameter for the kernel function (default: 1.0).
    epochs : int, optional
        The number of training epochs for the local model (default: 20).
    optim : str, optional
        The optimizer to use for training (default: "Adam").
    optim_paras : dict, optional
        Additional parameters for the optimizer (default: None).
    seed : int, optional
        Random seed for reproducibility (default: 42).
    verbose : bool, optional
        Whether to print training progress (default: True).
    device : str, optional
        The device to use for training ("cpu" or "gpu"). If "gpu" is specified but not available,
        an error is raised (default: None).

    Attributes
    ----------
    kernel : str or callable
        The kernel function used for computing weights.
    tau : float
        The bandwidth parameter for the kernel function.
    epochs : int
        The number of training epochs for the local model.
    optim : str
        The optimizer used for training.
    optim_paras : dict
        Additional parameters for the optimizer.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Whether to print training progress.
    device : str
        The device used for training ("cpu" or "cuda").
    X_ : ndarray, shape (n_samples, n_features)
        The training data.
    y_ : ndarray, shape (n_samples,)
        The target class labels for the training data.
    classes_ : ndarray, shape (n_classes,)
        The unique class labels.
    is_binary_ : bool
        Whether the classification task is binary.
    kernel_func_ : callable
        The resolved kernel function (either from `kernel_module` or the provided callable).
    _trainer : callable
        The method used to train the local model (binary or multiclass).
    """
    def __init__(self, kernel='gaussian', tau=1.0, epochs=20,
                 optim="Adam", optim_paras=None, seed=42, verbose=True, device=None):
        super().__init__()
        self.kernel = kernel
        self.tau = tau
        self.epochs = epochs
        self.optim = optim
        self.optim_paras = optim_paras
        self.seed = seed
        self.verbose = verbose
        if device == 'gpu':
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                raise ValueError("GPU is not available. Please set device to 'cpu'.")
        else:
            self.device = "cpu"

        self.network, self.optimizer = None, None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        self.is_binary_ = len(self.classes_) == 2
        if self.is_binary_:
            self._trainer = self._train_binary
        else:
            self._trainer = self._train_multiclass

        if self.optim_paras is None:
            self.optim_paras = {}

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        if isinstance(self.kernel, str):
            self.kernel_func_ = getattr(kernel_module, f"{self.kernel}_kernel")
        elif callable(self.kernel):
            self.kernel_func_ = self.kernel
        else:
            raise ValueError("kernel must be a string or callable")
        return self

    def _train_binary(self, X_tensor, y_train, sample_weights, x_query):
        y_tensor = torch.tensor((y_train == self.classes_[1]).astype(float).reshape(-1, 1), dtype=torch.float32, device=self.device)
        model = nn.Linear(X_tensor.shape[1], 1, bias=False).to(self.device)
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        optimizer = getattr(torch.optim, self.optim)(model.parameters(), **self.optim_paras)

        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = model(X_tensor)
            loss = criterion(logits, y_tensor)
            weighted_loss = (loss.reshape(-1, 1) * sample_weights).mean()
            weighted_loss.backward()
            optimizer.step()

        x_aug = torch.tensor(np.insert(x_query, 0, 1).reshape(1, -1), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = model(x_aug)
            prob = torch.sigmoid(logits).cpu().numpy().flatten()
            return np.stack([1 - prob, prob], axis=1)

    def _train_multiclass(self, X_tensor, y_train, sample_weights, x_query):
        y_tensor = torch.tensor(np.eye(len(self.classes_))[np.searchsorted(self.classes_, y_train)], dtype=torch.float32, device=self.device)
        model = nn.Linear(X_tensor.shape[1], len(self.classes_), bias=False).to(self.device)
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = getattr(torch.optim, self.optim)(model.parameters(), **self.optim_paras)

        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = model(X_tensor)
            loss = criterion(logits, y_tensor.argmax(dim=1))
            weighted_loss = (loss.reshape(-1, 1) * sample_weights).mean()
            weighted_loss.backward()
            optimizer.step()

        x_aug = torch.tensor(np.insert(x_query, 0, 1).reshape(1, -1), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = model(x_aug)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            return prob

    def _train_local_model(self, x_query):
        X_train = self.X_
        y_train = self.y_
        distances = np.linalg.norm(X_train - x_query, axis=1)
        weights = self.kernel_func_(distances, self.tau)

        X_tensor = torch.tensor(np.hstack([np.ones((X_train.shape[0], 1)), X_train]), dtype=torch.float32, device=self.device)
        sample_weights = torch.tensor(weights.reshape(-1, 1), dtype=torch.float32, device=self.device)
        return self._trainer(X_tensor, y_train, sample_weights, x_query)

    def predict_proba(self, X):
        """
        Predict the class probabilities for the input samples.
        """
        check_is_fitted(self, ['X_', 'y_', 'classes_', 'is_binary_'])
        X = check_array(X)
        probas = [self._train_local_model(x) for x in X]
        return np.vstack(probas)

    def predict(self, X):
        """
        Predict the class labels for the input samples.
        """
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

    def score(self, X, y):
        """
        Returns the accuracy of the model on the given data and labels.
        """
        return accuracy_score(y, self.predict(X))

    def evaluate(self, y_true, y_pred, list_metrics=("AS", "RS")):
        """
        Evaluate the model using the specified metrics.
        """
        return self._evaluate_cls(y_true=y_true, y_pred=y_pred, list_metrics=list_metrics)

    def scores(self, X, y, list_metrics=("AS", "RS")):
        """
        Compute the evaluation metrics for the model on the given data and labels.
        """
        y_pred = self.predict(X)
        return self.evaluate(y, y_pred, list_metrics)


class GdLwRegressor(BaseModel, RegressorMixin):
    """
    Gradient-Descent-based Locally Weighted Regressor.

    This class implements a locally weighted regression model using gradient descent
    optimization. It predicts target values by training a local model for each query point
    with a specified kernel function and bandwidth parameter.

    Parameters
    ----------
    kernel : str or callable, optional
        The kernel function to use. If a string is provided, it should match the name of a kernel
        function in the `kernel_module`. If a callable is provided, it should accept distances
        and `tau` as arguments and return weights.
    tau : float, optional
        The bandwidth parameter for the kernel function (default: 1.0).
    epochs : int, optional
        The number of training epochs for the local model (default: 20).
    optim : str, optional
        The optimizer to use for training (default: "Adam").
    optim_paras : dict, optional
        Additional parameters for the optimizer (default: None).
    seed : int, optional
        Random seed for reproducibility (default: 42).
    verbose : bool, optional
        Whether to print training progress (default: True).
    device : str, optional
        The device to use for training ("cpu" or "gpu"). If "gpu" is specified but not available,
        an error is raised (default: None).

    Attributes
    ----------
    kernel : str or callable
        The kernel function used for computing weights.
    tau : float
        The bandwidth parameter for the kernel function.
    epochs : int
        The number of training epochs for the local model.
    optim : str
        The optimizer used for training.
    optim_paras : dict
        Additional parameters for the optimizer.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Whether to print training progress.
    device : str
        The device used for training ("cpu" or "cuda").
    X_ : ndarray, shape (n_samples, n_features)
        The training data.
    y_ : ndarray, shape (n_samples,)
        The target values for the training data.
    kernel_func_ : callable
        The resolved kernel function (either from `kernel_module` or the provided callable).
    """

    def __init__(self, kernel='gaussian', tau=1.0, epochs=20,
                 optim='Adam', optim_paras=None, seed=42, verbose=True, device=None):
        """
        Initialize the GdLwRegressor.

        Parameters
        ----------
        kernel : str or callable, optional
            The kernel function to use (default: "gaussian").
        tau : float, optional
            The bandwidth parameter for the kernel function (default: 1.0).
        epochs : int, optional
            The number of training epochs for the local model (default: 20).
        optim : str, optional
            The optimizer to use for training (default: "Adam").
        optim_paras : dict, optional
            Additional parameters for the optimizer (default: None).
        seed : int, optional
            Random seed for reproducibility (default: 42).
        verbose : bool, optional
            Whether to print training progress (default: True).
        device : str, optional
            The device to use for training ("cpu" or "gpu"). If "gpu" is specified but not available,
            an error is raised (default: None).
        """
        super().__init__()
        self.kernel = kernel
        self.tau = tau
        self.epochs = epochs
        self.optim = optim
        self.optim_paras = optim_paras
        self.seed = seed
        self.verbose = verbose
        if device == 'gpu':
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                raise ValueError("GPU is not available. Please set device to 'cpu'.")
        else:
            self.device = "cpu"

        self.network, self.optimizer = None, None

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
        self : GdLwRegressor
            The fitted model.
        """
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        if self.optim_paras is None:
            self.optim_paras = {}

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        if isinstance(self.kernel, str):
            self.kernel_func_ = getattr(kernel_module, f"{self.kernel}_kernel")
        elif callable(self.kernel):
            self.kernel_func_ = self.kernel
        else:
            raise ValueError("kernel must be a string or callable")
        return self

    def _train_local_model(self, x_query):
        X_train = self.X_
        y_train = self.y_

        distances = np.linalg.norm(X_train - x_query, axis=1)
        weights = self.kernel_func_(distances, self.tau)

        X_tensor = torch.tensor(np.hstack([np.ones((X_train.shape[0], 1)), X_train]), dtype=torch.float32, device=self.device)
        sample_weights = torch.tensor(weights.reshape(-1, 1), dtype=torch.float32, device=self.device)

        y_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32, device=self.device)
        model = nn.Linear(X_tensor.shape[1], 1, bias=False).to(self.device)
        criterion = nn.MSELoss(reduction='none')
        optimizer = getattr(torch.optim, self.optim)(model.parameters(), **self.optim_paras)

        for _ in range(self.epochs):
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = criterion(output, y_tensor)
            weighted_loss = (loss * sample_weights).mean()
            weighted_loss.backward()
            optimizer.step()

        x_aug = torch.tensor(np.insert(x_query, 0, 1).reshape(1, -1), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred = model(x_aug).cpu().numpy()
            return pred

    def predict(self, X):
        """
        Predict target values for the given input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        preds : ndarray, shape (n_samples,)
            The predicted target values.
        """
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        preds = [self._train_local_model(x) for x in X]
        return np.array(preds).flatten()

    def score(self, X, y):
        """
        Compute the R2 score for the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        y : array-like, shape (n_samples,)
            The true target values.

        Returns
        -------
        score : float
            The R2 score of the predictions.
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
