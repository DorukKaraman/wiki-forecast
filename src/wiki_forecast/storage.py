import argparse
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from wiki_forecast.ingest import RAW_FILE

DATA_REPO = "DorukKaraman/wiki-forecast-data"
RAW_IN_REPO = "raw/pageviews.parquet"


def create_data_repo() -> None:
    """Create the dataset repository on the Hub (does nothing if it exists)."""
    HfApi().create_repo(DATA_REPO, repo_type="dataset", exist_ok=True)
    print(f"Dataset repo ready: https://huggingface.co/datasets/{DATA_REPO}")


def upload(local_path: Path, path_in_repo: str, message: str) -> None:
    """Upload one file to the dataset repository as a new commit."""
    HfApi().upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=DATA_REPO,
        repo_type="dataset",
        commit_message=message,
    )


def download(path_in_repo: str) -> Path:
    """Download one file from the dataset repository; returns the local path."""
    return Path(hf_hub_download(repo_id=DATA_REPO, filename=path_in_repo, repo_type="dataset"))


def load_pageviews() -> pd.DataFrame:
    """The current pageview table, straight from the Hub."""
    return pd.read_parquet(download(RAW_IN_REPO))


def check_roundtrip() -> None:
    """Verify that the copy on the Hub matches the local file exactly."""
    local = pd.read_parquet(RAW_FILE)
    remote = load_pageviews()
    pd.testing.assert_frame_equal(local, remote)
    print(f"OK: Hub copy matches local file ({len(remote):,} rows, "
          f"{remote['article'].nunique()} articles)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync data with the Hugging Face Hub.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="create the dataset repository")
    sub.add_parser("push-raw", help="upload the local pageview table")
    sub.add_parser("check", help="verify the Hub copy matches the local file")

    args = parser.parse_args()
    if args.command == "init":
        create_data_repo()
    elif args.command == "push-raw":
        upload(RAW_FILE, RAW_IN_REPO, "Upload pageview table")
        print(f"Uploaded {RAW_FILE} to {DATA_REPO}/{RAW_IN_REPO}")
    else:
        check_roundtrip()