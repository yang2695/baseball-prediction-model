"""Download and clean the historical MLB game file."""

from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

# FiveThirtyEight's old CDN now serves HTML for some clients.
# This is a CSV mirror of its archived historical file.
SOURCE_URL = "https://datahub.io/fivethirtyeight/mlb-elo/_r/-/data/mlb_elo.csv"
DEFAULT_RAW_PATH = Path("data/raw/mlb_elo.csv")
REQUIRED_COLUMNS = {
    "date", "season", "playoff", "team1", "team2",
    "neutral", "score1", "score2", "elo_prob1",
}


def download_games(destination: Path = DEFAULT_RAW_PATH, *, refresh: bool = False) -> Path:
    """Cache the original CSV without adding a large third-party file to Git."""
    destination = Path(destination)
    if destination.is_file() and not refresh:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(SOURCE_URL, headers={"User-Agent": "baseball-prediction-model/1.0"})
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(request, timeout=90) as response, temporary.open("wb") as output:
            first_chunk = response.read(1024 * 1024)
            first_line = first_chunk.lstrip(bytes.fromhex("efbbbf")).split(b"\n", 1)[0]
            if not first_line.startswith(b"date,") or b"season" not in first_line:
                raise ValueError("The data source did not return the expected MLB CSV header")
            output.write(first_chunk)
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_games(
    path: Path,
    *,
    min_season: int = 2015,
    max_season: int = 2022,
) -> pd.DataFrame:
    """Return completed, non-tied regular-season games in chronological order."""
    games = pd.read_csv(path, low_memory=False)
    missing = REQUIRED_COLUMNS.difference(games.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    games = games.copy()
    regular_season = games["playoff"].isna() | games["playoff"].astype(str).str.strip().eq("")
    games = games.loc[regular_season]
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    for column in ("season", "neutral", "score1", "score2", "elo_prob1"):
        games[column] = pd.to_numeric(games[column], errors="coerce")
    games = games.dropna(
        subset=["date", "season", "team1", "team2", "score1", "score2", "elo_prob1"]
    )
    games = games.loc[
        games["season"].between(min_season, max_season)
        & games["score1"].ne(games["score2"])
        & games["team1"].ne(games["team2"])
    ].copy()
    games["season"] = games["season"].astype(int)
    games["neutral"] = games["neutral"].fillna(0).astype(int)
    games["home_win"] = (games["score1"] > games["score2"]).astype(int)
    if games.empty:
        raise ValueError("No completed regular-season games in the requested years")
    return games.sort_values("date", kind="stable").reset_index(drop=True)
