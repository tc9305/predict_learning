"""Problem 2 data: align GSPC / EURUSD / FIGRX and encode daily % changes.

Encoding (HW2.pdf): x1 = GSPC % change today, x2 = EURUSD % change today,
y = FIGRX % change TOMORROW.
"""
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / 'data'


def _yahoo(name: str, year: int, column: str) -> pd.Series:
    df = pd.read_csv(DATA / f'{name}_{year}.csv', parse_dates=['Date'])
    return df.set_index('Date')[column].sort_index().rename(name)  # GSPC file is newest-first


def _ubc(year: int, invert: bool) -> pd.Series:
    # UBC columns: julian day, M/D/YYYY, weekday, rate (EUR per 1 USD)
    df = pd.read_csv(DATA / f'EURUSD_{year}.csv', header=None,
                     names=['jd', 'Date', 'dow', 'rate'])
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    rate = df.set_index('Date')['rate'].sort_index()
    return (1 / rate if invert else rate).rename('EURUSD')


def load_prices(year: int, fund_column: str = 'Close', invert_fx: bool = False) -> pd.DataFrame:
    """Inner-join the three series on dates present in all of them (step 3 of the preprocessing note)."""
    return pd.concat([_yahoo('GSPC', year, 'Close'), _ubc(year, invert_fx),
                      _yahoo('FIGRX', year, fund_column)], axis=1, join='inner')


def encode(prices: pd.DataFrame) -> pd.DataFrame:
    """% changes; pair today's inputs with tomorrow's fund change. Drops first and last aligned day."""
    pct = 100 * prices.pct_change()
    out = pd.DataFrame({'x1': pct['GSPC'], 'x2': pct['EURUSD'],
                        'y': pct['FIGRX'].shift(-1)})
    return out.dropna()


def load_year(year: int, **kwargs) -> pd.DataFrame:
    return encode(load_prices(year, **kwargs))
