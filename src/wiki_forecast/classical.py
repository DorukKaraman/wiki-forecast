import logging

import numpy as np
import pandas as pd


def prophet_model(holidays: pd.DataFrame | None = None):
    """Create a Prophet forecaster, optionally with known event windows."""

    def forecast(train, horizon):
        from prophet import Prophet

        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

        history = pd.DataFrame({"ds": train.index, "y": train.to_numpy()})
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            holidays=holidays,
        )
        model.fit(history)

        future = model.make_future_dataframe(periods=horizon, include_history=False)
        return model.predict(future)["yhat"].to_numpy()

    return forecast


def statsforecast_model(make_model):
    """Turn a statsforecast model into our forecaster interface.

    `make_model` is a function that creates a fresh, untrained model.
    """

    def forecast(train, horizon):
        model = make_model()
        y = np.asarray(train, dtype=float)
        return model.forecast(y=y, h=horizon)["mean"]

    return forecast