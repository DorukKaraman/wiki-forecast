import numpy as np


def log_transformed(forecaster):
    """Wrap a forecaster so it trains on log-views and returns normal views."""

    def forecast(train, horizon):
        preds = forecaster(np.log1p(train), horizon)
        return np.expm1(preds)

    return forecast