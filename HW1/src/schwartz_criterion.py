"""Lecture Set 3, p. 34: multiplicative analytic Schwartz criterion."""
import numpy as np


def schwartz(training_mse: float, dof: int, n: int) -> float:
    if not 0 < dof < n:
        raise ValueError('Schwartz criterion requires 0 < DoF < n')
    if training_mse < 0 or not np.isfinite(training_mse):
        raise ValueError('MSE must be finite and nonnegative')
    p = dof / n
    return float((1 + p*np.log(n)/(1-p))*training_mse)
