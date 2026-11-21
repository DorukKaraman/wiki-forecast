import argparse
from datetime import date, timedelta

import pandas as pd

from wiki_forecast.ingest import RAW_FILE, fetch_many, fill_gaps
from wiki_forecast.storage import RAW_IN_REPO, load_pageviews, upload


def validate(old: pd.DataFrame, new: pd.DataFrame, max_lagging: float = 0.05) -> None:
    """Refuse to continue if the updated table looks wrong."""
    problems = []
    if new["article"].nunique() != old["article"].nunique():
        problems.append(
            f"article count changed: {old['article'].nunique()} -> {new['article'].nunique()}"
        )
    if len(new) <= len(old):
        problems.append(f"no new rows: {len(old):,} -> {len(new):,}")
    if new.duplicated(["date", "article"]).any():
        problems.append("duplicate (date, article) rows")
    if new["views"].isna().any() or (new["views"] < 0).any():
        problems.append("missing or negative view counts")

    # Completeness: every article should reach the newest date.
    last_by_article = new.groupby("article")["date"].max()
    lagging = last_by_article[last_by_article < last_by_article.max()]
    if len(lagging) > max_lagging * len(last_by_article):
        problems.append(
            f"{len(lagging)} articles not up to date (e.g. {', '.join(lagging.index[:5])})"
        )
    elif len(lagging):
        print(f"Warning: {len(lagging)} articles lag behind and will catch up next run: "
              f"{', '.join(lagging.index)}")

    if problems:
        raise ValueError("Validation failed, nothing uploaded:\n  " + "\n  ".join(problems))


def update_pageviews(upload_result: bool = True) -> pd.DataFrame:
    """Fetch the days each article is missing and (optionally) upload."""
    old = load_pageviews()
    yesterday = pd.Timestamp(date.today() - timedelta(days=1))

    # Each article continues from its OWN last day, so a failed article
    # simply catches up on the next run instead of leaving a gap.
    last_by_article = old.groupby("article")["date"].max()
    behind = last_by_article[last_by_article < yesterday]
    if behind.empty:
        print(f"Already up to date (last day {last_by_article.max().date()})")
        return old

    frames = []
    for last_day, group in behind.groupby(behind):
        start = (last_day + pd.Timedelta(days=1)).strftime("%Y%m%d")
        print(f"Fetching from {start} for {len(group)} articles")
        frames.append(fetch_many(sorted(group.index), start=start))
    new_rows = pd.concat(frames, ignore_index=True)
    if new_rows.empty:
        print("No new data published yet; nothing to do")
        return old

    combined = (
        pd.concat([old, new_rows], ignore_index=True)
        .drop_duplicates(["date", "article"], keep="last")
    )
    combined = fill_gaps(combined).sort_values(["article", "date"]).reset_index(drop=True)
    validate(old, combined)

    new_last = combined["date"].max().date()
    print(f"Validated: {len(combined) - len(old):,} new rows, data now through {new_last}")
    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(RAW_FILE, index=False)

    if upload_result:
        upload(RAW_FILE, RAW_IN_REPO, f"Daily update: pageviews through {new_last}")
        print("Uploaded to the Hub")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily pipeline steps.")
    sub = parser.add_subparsers(dest="command", required=True)
    update = sub.add_parser("update", help="fetch new days and upload the pageview table")
    update.add_argument("--dry-run", action="store_true", help="don't upload, only save locally")

    args = parser.parse_args()
    if args.command == "update":
        update_pageviews(upload_result=not args.dry_run)