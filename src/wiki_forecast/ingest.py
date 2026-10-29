from datetime import date
from urllib.parse import quote

import pandas as pd
import requests

API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"

HEADERS = {
    "User-Agent": "wiki-forecast/0.1 (https://github.com/DorukKaraman/wiki-forecast)"
}


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
    url = f"{API}/{project}/all-access/{agent}/{title}/daily/{start}/{end}"

    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    items = response.json()["items"]
    df = pd.DataFrame(items)
    df["date"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H")
    df["article"] = article
    return df[["date", "article", "views"]]


if __name__ == "__main__":
    df = fetch_pageviews("Oktoberfest")
    df.to_parquet("data/raw/oktoberfest.parquet", index=False)

    print(df.head())
    print(df.tail())
    print(f"Rows: {len(df)}")