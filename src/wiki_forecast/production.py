import argparse
import json
import tempfile
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from wiki_forecast.features import FEATURES, FEATURES_PEAK, build_dataset, to_log_wide, to_views
from wiki_forecast.global_model import PARAMS, fit, make_X, quantile_params
from wiki_forecast.storage import DATA_REPO, load_pageviews

MODEL_REPO = "DorukKaraman/wiki-forecast-model"
FEATURES_MAIN = [*FEATURES, "season_corr", "article", *FEATURES_PEAK]
HORIZONS = range(1, 29)
QUANTILE_PARAMS = {"q10": quantile_params(0.1), "q50": PARAMS, "q90": quantile_params(0.9)}
RESULTS_DIR = Path("data/results")


# --- calibration -------------------------------------------------------------

def calibration_from_backtest(window: int = 12) -> dict[int, float]:
    """Conformal correction per horizon, from the latest backtest folds (notebook 06)."""
    keys = ["article", "cutoff", "date"]
    q50 = pd.read_parquet(RESULTS_DIR / "lgbm_global_corr_id_peak.parquet")
    q10 = pd.read_parquet(RESULTS_DIR / "lgbm_main_peak_q10.parquet")
    q90 = pd.read_parquet(RESULTS_DIR / "lgbm_main_peak_q90.parquet")
    iv = (
        q50.rename(columns={"pred": "q50"})
        .merge(q10[[*keys, "pred"]].rename(columns={"pred": "q10"}), on=keys)
        .merge(q90[[*keys, "pred"]].rename(columns={"pred": "q90"}), on=keys)
    )
    iv[["q10", "q50", "q90"]] = np.sort(iv[["q10", "q50", "q90"]].to_numpy(), axis=1)
    iv["score"] = np.maximum(
        np.log1p(iv["q10"]) - np.log1p(iv["actual"]),
        np.log1p(iv["actual"]) - np.log1p(iv["q90"]),
    )
    latest = sorted(iv["cutoff"].unique())[-window:]
    q_by_step = iv[iv["cutoff"].isin(latest)].groupby("step")["score"].quantile(0.8)
    return {int(step): float(value) for step, value in q_by_step.items()}


# --- training and publishing -------------------------------------------------

def train_production(df: pd.DataFrame) -> tuple[dict, dict]:
    """Train q10 / q50 / q90 on all available data."""
    log_wide = to_log_wide(df)
    last_day = log_wide.index[-1]
    articles = list(log_wide.columns)

    origins = pd.date_range("2018-01-01", last_day, freq="14D")
    data = build_dataset(log_wide, origins, HORIZONS)
    train = data[(data["target_date"] <= last_day) & data["target"].notna()]
    print(f"Training on {len(train):,} rows through {last_day.date()}")

    models = {}
    for name, params in QUANTILE_PARAMS.items():
        models[name] = fit(train, FEATURES_MAIN, articles, params)
        print(f"  trained {name}")

    meta = {
        "version": f"v{last_day.date()}",
        "trained_through": str(last_day.date()),
        "trained_on": str(date.today()),
        "data_repo": DATA_REPO,
        "data_revision": HfApi().repo_info(DATA_REPO, repo_type="dataset").sha,
        "features": FEATURES_MAIN,
        "articles": articles,
        "horizons": list(HORIZONS),
        "params": QUANTILE_PARAMS,
        "n_training_rows": int(len(train)),
    }
    return models, meta


def publish(models: dict, meta: dict, calibration: dict[int, float]) -> None:
    """Upload models + metadata + calibration as one commit, then tag the version."""
    api = HfApi()
    api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, model in models.items():
            model.booster_.save_model(str(tmp / f"{name}.txt"))
        (tmp / "metadata.json").write_text(json.dumps(meta, indent=2))
        (tmp / "calibration.json").write_text(json.dumps(calibration, indent=2))
        api.upload_folder(
            folder_path=str(tmp),
            repo_id=MODEL_REPO,
            repo_type="model",
            commit_message=f"Production models trained through {meta['trained_through']}",
        )
    api.create_tag(MODEL_REPO, tag=meta["version"], repo_type="model", exist_ok=True)
    print(f"Published {meta['version']}: https://huggingface.co/{MODEL_REPO}")


# --- loading (used by the daily forecast job) --------------------------------

def load_production(revision: str | None = None) -> tuple[dict, dict, dict[int, float]]:
    """Load models, metadata and calibration from the Hub (latest, or a given tag)."""
    def get(filename: str) -> Path:
        return Path(hf_hub_download(MODEL_REPO, filename, revision=revision))

    models = {name: lgb.Booster(model_file=str(get(f"{name}.txt"))) for name in QUANTILE_PARAMS}
    meta = json.loads(get("metadata.json").read_text())
    calibration = {int(k): v for k, v in json.loads(get("calibration.json").read_text()).items()}
    return models, meta, calibration


def make_forecasts(
    df: pd.DataFrame, models: dict, meta: dict, calibration: dict[int, float]
) -> pd.DataFrame:
    """Forecast every article 28 days ahead from the latest date in `df`."""
    log_wide = to_log_wide(df)
    origin = log_wide.index[-1]
    rows = build_dataset(log_wide, [origin], meta["horizons"])

    X = make_X(rows, meta["features"], meta["articles"])
    level = rows["level"].to_numpy(np.float64)
    preds = np.column_stack(
        [to_views(models[name].predict(X), level) for name in ("q10", "q50", "q90")]
    )
    preds = np.sort(preds, axis=1)  # fix any quantile crossing

    # Conformal widening per horizon (in log space), as in the backtest.
    adjust = rows["horizon"].map(calibration).to_numpy(np.float64)
    q10 = np.expm1(np.log1p(preds[:, 0]) - adjust).clip(min=0)
    q90 = np.expm1(np.log1p(preds[:, 2]) + adjust)

    return pd.DataFrame({
        "article": rows["article"].to_numpy(),
        "origin": origin,
        "horizon": rows["horizon"].to_numpy(),
        "date": rows["target_date"].to_numpy(),
        "q10": q10,
        "q50": preds[:, 1],
        "q90": q90,
        "model_version": meta["version"],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production model management.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="train on all data and publish to the Hub")
    sub.add_parser("check", help="download the published models and print their metadata")

    args = parser.parse_args()
    if args.command == "train":
        models, meta = train_production(load_pageviews())
        publish(models, meta, calibration_from_backtest())
    else:
        models, meta, calibration = load_production()
        print(f"{meta['version']}: {len(meta['features'])} features, "
              f"{len(meta['articles'])} articles, trained on {meta['n_training_rows']:,} rows")
        print(f"Data revision: {meta['data_revision'][:10]}")
        print(f"Calibration day 1: {calibration[1]:+.3f}, day 28: {calibration[28]:+.3f}")