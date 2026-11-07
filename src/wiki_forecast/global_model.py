import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from wiki_forecast.features import FEATURES, build_dataset, to_views

# Chosen up front, not tuned on the backtest.
PARAMS = {
    "objective": "l1",        # median in log space = median in views
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}


def make_X(df: pd.DataFrame, features: list[str], articles: list[str]) -> pd.DataFrame:
    """Select model inputs; encode `article` with a fixed category list if it's used."""
    X = df[features]
    if "article" in features:
        X = X.assign(article=pd.Categorical(df["article"], categories=articles))
    return X


def fit(train: pd.DataFrame, features: list[str], articles: list[str], params=PARAMS):
    model = lgb.LGBMRegressor(**params)
    model.fit(make_X(train, features, articles), train["target"])
    return model


def global_backtest(
    log_wide: pd.DataFrame,
    train_data: pd.DataFrame,
    cutoffs,
    horizons,
    name: str,
    retrain_every: int = 6,
    features=FEATURES,
) -> pd.DataFrame:
    """Backtest one model across all articles in `log_wide`.

    For each cutoff: (re)train on rows whose TARGET date is <= cutoff,
    then forecast all articles from the cutoff. Same output format as
    `evaluation.backtest`, plus an `article` column.
    """
    cutoffs = pd.DatetimeIndex(cutoffs)
    articles = list(log_wide.columns)
    predict_rows = build_dataset(log_wide, cutoffs, horizons)

    # MASE denominator per article: mean |y_t - y_(t-7)| up to each date.
    raw = np.expm1(log_wide)
    scale = raw.diff(7).abs().expanding().mean()

    results, model = [], None
    for i, cutoff in enumerate(cutoffs):
        if i % retrain_every == 0:
            start = time.perf_counter()
            train = train_data[(train_data["target_date"] <= cutoff) & train_data["target"].notna()]
            model = fit(train, features, articles)
            print(f"{name}: trained on {len(train):,} rows up to {cutoff.date()} "
                  f"in {time.perf_counter() - start:.0f}s")

        rows = predict_rows[predict_rows["origin"] == cutoff]
        level = rows["level"].to_numpy(np.float64)
        pred = model.predict(make_X(rows, features, articles))

        results.append(pd.DataFrame({
            "model": name,
            "article": rows["article"].to_numpy(),
            "cutoff": cutoff,
            "date": rows["target_date"].to_numpy(),
            "step": rows["horizon"].to_numpy(),
            "actual": to_views(rows["target"].to_numpy(np.float64), level),
            "pred": to_views(pred, level),
            "scale": scale.loc[cutoff, rows["article"]].to_numpy(),
        }))
    return pd.concat(results, ignore_index=True)