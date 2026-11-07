import argparse
import time
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

API = "https://wikimedia.org/api/rest_v1/metrics/pageviews"

# Wikimedia asks for a descriptive User-Agent with contact info.
# A project URL is an accepted alternative to an email address.
HEADERS = {
    "User-Agent": "wiki-forecast/0.1 (https://github.com/DorukKaraman/wiki-forecast)"
}

ARTICLES_FILE = Path("configs/articles.txt")
RAW_FILE = Path("data/raw/pageviews.parquet")
EXCLUDED_FILE = Path("configs/excluded.txt")

# Pages in top lists that aren't real articles.
EXCLUDED_PREFIXES = ("Special:", "Wikipedia:", "File:", "Portal:", "Help:", "Talk:", "User:", "Category:", "Template:")
EXCLUDED_TITLES = {"Main_Page", "-"}

session = requests.Session()
session.headers.update(HEADERS)


def _get(url: str, retries: int = 3) -> dict:
    """GET with retries for rate limiting (429) and server errors (5xx)."""
    for attempt in range(retries):
        response = session.get(url, timeout=30)
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(2**attempt)  # wait 1s, 2s, 4s
            continue
        response.raise_for_status()
        return response.json()
    response.raise_for_status()
    return response.json()


def fetch_pageviews(
    article: str,
    project: str = "en.wikipedia",
    start: str = "20150701",
    end: str | None = None,
    agent: str = "user",
) -> pd.DataFrame:
    """Fetch the daily pageviews of one Wikipedia article."""
    end = end or date.today().strftime("%Y%m%d")
    title = quote(article.replace(" ", "_"), safe="")
    url = f"{API}/per-article/{project}/all-access/{agent}/{title}/daily/{start}/{end}"

    df = pd.DataFrame(_get(url)["items"])
    df["date"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H")
    df["article"] = article
    return df[["date", "article", "views"]]


def top_articles(year: int, month: int, project: str = "en.wikipedia") -> list[str]:
    """The ~1,000 most-viewed articles of one month."""
    url = f"{API}/top/{project}/all-access/{year}/{month:02d}/all-days"
    articles = _get(url)["items"][0]["articles"]
    return [a["article"] for a in articles]


def is_real_article(title: str) -> bool:
    return title not in EXCLUDED_TITLES and not title.startswith(EXCLUDED_PREFIXES)


def read_list(path: Path) -> list[str]:
    """Read one title per line, ignoring blank lines and # comments."""
    if not path.exists():
        return []
    lines = (line.strip() for line in path.read_text().splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def select_articles(months: list[tuple[int, int]], n: int) -> list[str]:
    """Articles that appear in the most monthly top lists (consistently popular)."""
    excluded = set(read_list(EXCLUDED_FILE))
    counts = Counter()
    used = 0
    for year, month in months:
        try:
            titles = top_articles(year, month)
        except requests.HTTPError as error:
            print(f"  skipped {year}-{month:02d}: HTTP {error.response.status_code}")
            continue
        counts.update(t for t in titles if is_real_article(t) and t not in excluded)
        used += 1
        time.sleep(0.1)  # be polite to the API

    print(f"Used {used} of {len(months)} monthly top lists")
    selected = [title for title, _ in counts.most_common(n)]
    if "Oktoberfest" not in selected:
        selected = ["Oktoberfest", *selected[:-1]]
    return selected


def fetch_many(articles: list[str]) -> pd.DataFrame:
    """Fetch every article; skip (and report) the ones that fail."""
    frames, failed = [], []
    for i, article in enumerate(articles, start=1):
        try:
            frames.append(fetch_pageviews(article))
        except requests.HTTPError as error:
            failed.append(article)
            print(f"  skipped {article}: {error}")
        if i % 50 == 0:
            print(f"  {i}/{len(articles)} fetched")
        time.sleep(0.05)
    print(f"Done: {len(frames)} fetched, {len(failed)} skipped")
    return pd.concat(frames, ignore_index=True)


def fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Give every article a continuous daily index from its first day; missing days = 0."""
    filled = []
    for article, g in df.groupby("article"):
        full_range = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        g = g.set_index("date").reindex(full_range).rename_axis("date").reset_index()
        g["article"] = article
        g["views"] = g["views"].fillna(0).astype(int)
        filled.append(g)
    return pd.concat(filled, ignore_index=True)


def keep_long_history(df: pd.DataFrame, min_years: int = 3) -> pd.DataFrame:
    """Keep articles whose data starts at least `min_years` before the last day."""
    cutoff = df["date"].max() - pd.DateOffset(years=min_years)
    first_day = df.groupby("article")["date"].min()
    keep = first_day[first_day <= cutoff].index
    return df[df["article"].isin(keep)]


def cmd_select(n: int) -> None:
    months = [(year, month) for year in (2023, 2024, 2025) for month in range(1, 13)]
    articles = select_articles(months, n)
    ARTICLES_FILE.parent.mkdir(parents=True, exist_ok=True)
    ARTICLES_FILE.write_text("\n".join(articles) + "\n")
    print(f"Wrote {len(articles)} articles to {ARTICLES_FILE}")


def cmd_fetch() -> None:
    excluded = set(read_list(EXCLUDED_FILE))
    articles = [a for a in read_list(ARTICLES_FILE) if a not in excluded]
    print(f"{len(articles)} articles to fetch ({len(excluded)} excluded)")

    df = fetch_many(articles)
    df = keep_long_history(fill_gaps(df))

    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_FILE, index=False)
    print(
        f"Saved {df['article'].nunique()} articles, {len(df):,} rows "
        f"({df['date'].min().date()} to {df['date'].max().date()}) to {RAW_FILE}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Wikipedia pageviews.")
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select", help="choose articles from monthly top lists")
    select.add_argument("--n", type=int, default=600)
    sub.add_parser("fetch", help="download the daily history of every chosen article")

    args = parser.parse_args()
    if args.command == "select":
        cmd_select(args.n)
    else:
        cmd_fetch()