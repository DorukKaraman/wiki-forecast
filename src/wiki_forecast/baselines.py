import numpy as np
import pandas as pd


def naive(train: pd.Series, horizon: int) -> np.ndarray:
    """Every future day equals the last observed day."""
    return np.repeat(train.iloc[-1], horizon)


def seasonal_naive(season: int):
    """Every future day equals the value exactly `season` days earlier."""

    def forecast(train: pd.Series, horizon: int) -> np.ndarray:
        last_season = train.iloc[-season:].to_numpy()
        repeats = int(np.ceil(horizon / season))
        return np.tile(last_season, repeats)[:horizon]

    return forecast