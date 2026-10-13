"""OLS in the exact cosine or monomial basis specified by the assignment."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .schwartz_criterion import schwartz

FAMILIES = ('trigonometric', 'algebraic')


def design_matrix(x, m: int, family: str):
    if not isinstance(m, (int, np.integer)) or m < 0:
        raise ValueError('m must be a nonnegative integer')
    x = np.asarray(x, dtype=float).reshape(-1)
    if family == 'trigonometric':
        return np.column_stack([np.ones(len(x)), np.cos(2*np.pi*x[:, None]*np.arange(1, m+1))])
    if family == 'algebraic':
        return np.vander(x, N=m+1, increasing=True)
    raise ValueError(f'Unknown family: {family}')


@dataclass
class PolynomialModel:
    family: str
    m: int
    coefficients: np.ndarray
    rank: int
    condition_number: float

    def predict(self, x):
        return design_matrix(x, self.m, self.family) @ self.coefficients


def fit_polynomial(x, y, m: int, family: str) -> PolynomialModel:
    a = design_matrix(x, m, family)
    y = np.asarray(y, dtype=float)
    if len(y) != len(a) or not np.isfinite(a).all() or not np.isfinite(y).all():
        raise ValueError('Invalid fitting observations')
    # SVD-based least squares avoids explicitly forming (A.T @ A)^(-1).
    coefficients, _, rank, singular = np.linalg.lstsq(a, y, rcond=None)
    condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else float('inf')
    return PolynomialModel(family, m, coefficients, int(rank), condition)


def select_polynomial(x, y, family: str):
    """Use only these training observations; exact score ties favor smaller m."""
    n = len(x)
    if n < 2:
        raise ValueError('At least two training observations required')
    models, rows = [], []
    for m in range(n-1):  # DoF=m+1 < n
        model = fit_polynomial(x, y, m, family)
        mse = float(np.mean((np.asarray(y)-model.predict(x))**2))
        rows.append({'m': m, 'DoF': m+1, 'n_train': n, 'p': (m+1)/n,
                     'training_mse': mse, 'schwartz_criterion': schwartz(mse, m+1, n),
                     'rank': model.rank, 'condition_number': model.condition_number,
                     'ill_conditioned': model.condition_number > 1e8})
        models.append(model)
    table = pd.DataFrame(rows)
    best = int(np.argmin(table.schwartz_criterion.to_numpy()))
    return models[best], table
