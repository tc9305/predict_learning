"""Validate the supplied observations without changing their labels."""
from pathlib import Path
import numpy as np
import pandas as pd

COLUMNS = ['State', 'Obesity', 'Election_Result', 'State_Code']


def load_election(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, header=None, names=COLUMNS, encoding='utf-8-sig')
    if len(data) != 51 or data.isna().any().any():
        raise ValueError(f'{path.name}: expected 51 complete observations')
    for column in ['State', 'State_Code']:
        if data[column].duplicated().any():
            raise ValueError(f'{path.name}: duplicate {column}')
    if set(data.Election_Result) != {'D', 'R'}:
        raise ValueError(f'{path.name}: invalid election labels')
    if data.State_Code.eq('DC').sum() != 1 or data.State.eq('Alabama').sum() != 1:
        raise ValueError(f'{path.name}: missing Alabama or DC')
    text = data.Obesity.astype(str)
    if not text.str.fullmatch(r'\d+(?:\.\d+)?%').all():
        raise ValueError(f'{path.name}: invalid percentage')
    data['Obesity'] = pd.to_numeric(text.str.removesuffix('%'), errors='raise')
    if not np.isfinite(data.Obesity).all() or not data.Obesity.between(0, 100).all():
        raise ValueError(f'{path.name}: invalid obesity range')
    return data


def election_summary(data: pd.DataFrame, year: int) -> dict:
    return {'year': year, 'n': len(data), 'D': int(data.Election_Result.eq('D').sum()),
            'R': int(data.Election_Result.eq('R').sum()),
            **{f'obesity_{k}': v for k, v in data.Obesity.describe().items()}}


def generate_regression(seed: int = 42) -> pd.DataFrame:
    """A new reproducible sample, not the original textbook observations."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, 10)
    noise = rng.normal(0, 0.5, 10)  # variance 0.25
    return pd.DataFrame({'sample_id': np.arange(10), 'x': x,
                         'y': x**2 + 0.1*x + noise,
                         'true_function': x**2 + 0.1*x, 'noise': noise})
