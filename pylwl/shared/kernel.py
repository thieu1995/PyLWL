#!/usr/bin/env python
# Created by "Thieu" at 13:14, 16/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np


def gaussian_kernel(distances, tau=1.0):
    """Gaussian kernel: exp(-||x - xi||^2 / (2 * tau^2))"""
    weights = np.exp(- (distances ** 2) / (2 * tau ** 2))
    return weights


def tricube_kernel(distances, tau=1.0):
    """Tricube kernel: (1 - |u|^3)^3 if |u| <= 1 else 0"""
    u = distances / tau
    mask = np.abs(u) < 1
    weights = np.zeros_like(distances)
    weights[mask] = (1 - np.abs(u[mask]) ** 3) ** 3
    return weights


def epanechnikov_kernel(distances, tau=1.0):
    """Epanechnikov kernel: 0.75 * (1 - u^2) if |u| <= 1 else 0"""
    u = distances / tau
    mask = np.abs(u) < 1
    weights = np.zeros_like(distances)
    weights[mask] = 0.75 * (1 - u[mask] ** 2)
    return weights


def uniform_kernel(distances, tau=1.0):
    """Uniform kernel: 1 if |u| <= 1 else 0"""
    u = distances / tau
    return (np.abs(u) <= 1).astype(float)


def cosine_kernel(distances, tau=1.0):
    """Cosine kernel: cos(pi * u / 2) if |u| <= 1 else 0"""
    u = distances / tau
    mask = np.abs(u) < 1
    weights = np.zeros_like(distances)
    weights[mask] = np.cos(np.pi * u[mask] / 2)
    return weights
