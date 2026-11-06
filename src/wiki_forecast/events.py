import pandas as pd

# The festival was cancelled in these years (COVID).
CANCELLED_YEARS = {2020, 2021}


def oktoberfest_dates(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Start and end date of the Munich Oktoberfest in a given year."""
    sep15 = pd.Timestamp(year, 9, 15)
    days_to_saturday = (5 - sep15.weekday()) % 7 or 7  # Saturday = 5, strictly after the 15th
    start = sep15 + pd.Timedelta(days=days_to_saturday)

    oct1 = pd.Timestamp(year, 10, 1)
    first_sunday = oct1 + pd.Timedelta(days=(6 - oct1.weekday()) % 7)  # Sunday = 6
    end = max(first_sunday, pd.Timestamp(year, 10, 3))
    return start, end


def oktoberfest_holidays(
    first_year: int = 2015,
    last_year: int = 2030,
    build_up_days: int = 14,
    decay_days: int = 7,
) -> pd.DataFrame:
    """Festival windows in Prophet's holidays format."""
    rows = []
    for year in range(first_year, last_year + 1):
        if year in CANCELLED_YEARS:
            continue
        start, end = oktoberfest_dates(year)
        rows.append(
            {
                "holiday": "oktoberfest",
                "ds": start,
                "lower_window": -build_up_days,  # days before opening with their own effect
                "upper_window": (end - start).days + decay_days,  # festival plus aftermath
            }
        )
    return pd.DataFrame(rows)