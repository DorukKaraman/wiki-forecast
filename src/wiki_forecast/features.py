import numpy as np
import pandas as pd

LAGS = [0, 1, 2, 3, 6, 13, 27]
ROLL_MEANS = [28, 91, 364]
ROLL_STDS = [7, 28]

FEATURES = [
    "level",
    *[f"lag_{k}" for k in LAGS],
    *[f"mean_{w}" for w in ROLL_MEANS],
    *[f"std_{w}" for w in ROLL_STDS],
    "season_1y",
    "season_2y",
    "horizon",
    "dow",
    "doy",
    "month",
]

FEATURES_WITH_CORR = [*FEATURES, "season_corr"]


def to_log_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Long table (date, article, views) -> wide table of log1p(views), one column per article."""
    wide = df.pivot(index="date", columns="article", values="views").asfreq("D")
    return np.log1p(wide)


def build_dataset(log_wide: pd.DataFrame, origins, horizons) -> pd.DataFrame:
    """One row per (article, forecast origin, horizon).

    Every feature uses only data up to and including the origin date.
    The target (log-views at origin + horizon, relative to level) is NaN
    where the future isn't available.
    """
    origins = pd.DatetimeIndex(origins)

    def at_origins(frame: pd.DataFrame) -> pd.Series:
        """Values of a wide frame on the origin dates, as a (article, origin) series."""
        return (
            frame.reindex(origins)
            .rename_axis(index="origin", columns="article")
            .unstack()
        )

    # --- features known at the origin ------------------------------------
    level = log_wide.rolling(7).mean()
    base = {"level": at_origins(level)}
    for k in LAGS:
        base[f"lag_{k}"] = at_origins(log_wide.shift(k) - level)
    for w in ROLL_MEANS:
        base[f"mean_{w}"] = at_origins(log_wide.rolling(w).mean() - level)
    for w in ROLL_STDS:
        base[f"std_{w}"] = at_origins(log_wide.rolling(w).std())
    # How reliably does this series repeat last year's movements?
    # Correlation between 28-day changes and the same changes one year earlier,
    change = log_wide - log_wide.shift(28)
    season_corr = change.rolling(728, min_periods=364).corr(change.shift(364), pairwise=False)
    base["season_corr"] = at_origins(season_corr.clip(-1, 1))
    base = pd.DataFrame(base).dropna(subset=["level"])

    # --- horizon-specific features and target ----------------------------
    smooth = log_wide.rolling(7, center=True).mean()
    frames = []
    for h in horizons:
        f = base.copy()
        f["horizon"] = h
        for years in (1, 2):
            # value around (target date - years*364), as seen from the origin
            shifted = smooth.shift(364 * years - h)
            f[f"season_{years}y"] = at_origins(shifted).reindex(f.index) - f["level"]
        f["target"] = at_origins(log_wide.shift(-h)).reindex(f.index) - f["level"]
        frames.append(f)

    data = pd.concat(frames).reset_index()

    # --- calendar of the target date -------------------------------------
    data["target_date"] = data["origin"] + pd.to_timedelta(data["horizon"], unit="D")
    data["dow"] = data["target_date"].dt.day_of_week
    data["doy"] = data["target_date"].dt.day_of_year
    data["month"] = data["target_date"].dt.month

    # float32 halves the memory, and the precision is plenty for this
    float_cols = data.select_dtypes("float64").columns
    data[float_cols] = data[float_cols].astype("float32")
    return data


def to_views(pred_relative: np.ndarray, level: np.ndarray) -> np.ndarray:
    """Turn a relative log prediction back into views."""
    return np.expm1(pred_relative + level)