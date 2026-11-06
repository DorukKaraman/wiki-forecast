from collections.abc import Callable

import numpy as np
import pandas as pd

# A forecaster is any function that takes the training history and a
# horizon, and returns that many predictions. Every method we try
# (baselines, ETS, LightGBM, ...) will follow this same interface.
Forecaster = Callable[[pd.Series, int], np.ndarray]


def make_cutoffs(
    series: pd.Series, horizon: int, n_folds: int, step: int
) -> list[pd.Timestamp]:
    """Cutoff dates so that the last fold ends on the last available day."""
    last_cutoff = series.index[-1] - pd.Timedelta(days=horizon)
    return [
        last_cutoff - pd.Timedelta(days=step * i) for i in reversed(range(n_folds))
    ]


def backtest(
    series: pd.Series,
    forecaster: Forecaster,
    name: str,
    horizon: int = 28,
    n_folds: int = 26,
    step: int = 28,
    season: int = 7,
) -> pd.DataFrame:
    """Rolling-origin backtest. Returns one row per forecasted day."""
    folds = []
    for cutoff in make_cutoffs(series, horizon, n_folds, step):
        train = series[:cutoff]  # everything up to and including the cutoff
        test = series[cutoff + pd.Timedelta(days=1) : cutoff + pd.Timedelta(days=horizon)]

        preds = np.asarray(forecaster(train, horizon))
        assert len(preds) == horizon, f"{name} returned {len(preds)} predictions"

        # MASE denominator: in-sample error of "same day last week", per fold.
        scale = train.diff(season).abs().mean()

        folds.append(
            pd.DataFrame(
                {
                    "model": name,
                    "cutoff": cutoff,
                    "date": test.index,
                    "step": np.arange(1, horizon + 1),
                    "actual": test.to_numpy(),
                    "pred": preds,
                    "scale": scale,
                }
            )
        )
    return pd.concat(folds, ignore_index=True)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """MAE and MASE per model, best model first."""
    r = results.assign(abs_err=(results["actual"] - results["pred"]).abs())
    r["scaled_err"] = r["abs_err"] / r["scale"]
    return (
        r.groupby("model")
        .agg(MAE=("abs_err", "mean"), MASE=("scaled_err", "mean"))
        .sort_values("MASE")
    )