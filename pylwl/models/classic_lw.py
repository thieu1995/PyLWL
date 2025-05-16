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

