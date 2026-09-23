"""Download completed regular-season games from the public MLB Stats API."""

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
DEFAULT_RAW_PATH = Path("data/raw/mlb_games.csv")
REQUIRED_COLUMNS = {
    "date", "season", "team1", "team2", "neutral", "score1", "score2",
}


def parse_schedule(payload: dict, season: int) -> list[dict]:
    """Normalize a season schedule without assuming unplayed games have scores."""
    if not isinstance(payload.get("dates"), list):
        raise ValueError(f"MLB response for {season} is missing dates")
    rows = []
    seen = set()
    for day in payload["dates"]:
        for game in day.get("games", []):
            if game.get("gameType") != "R":
                continue
            if game.get("status", {}).get("abstractGameState") != "Final":
                continue
            home = game["teams"]["home"]
            away = game["teams"]["away"]
            if "score" not in home or "score" not in away:
                continue
            pk = int(game["gamePk"])
            if pk in seen:
                continue
            seen.add(pk)
            rows.append({
                "game_id": pk,
                "date": game.get("officialDate", day["date"]),
                "season": season,
                "team1": str(home["team"]["id"]),
                "team2": str(away["team"]["id"]),
                "neutral": int(bool(game.get("neutralSite", False))),
                "score1": int(home["score"]),
                "score2": int(away["score"]),
            })
    return rows


def download_games(
    destination: Path = DEFAULT_RAW_PATH,
    *,
    refresh: bool = False,
    start_season: int = 2015,
    end_season: int = 2025,
) -> Path:
    """Fetch all seasons, then atomically cache a normalized CSV.

    Reject incomplete historical seasons rather than silently backtesting on a
    partial or broken download. 2020 had an intentionally shortened schedule.
    """
    destination = Path(destination)
    if destination.is_file() and not refresh:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    rows = []
    try:
        for season in range(start_season, end_season + 1):
            params = urlencode({"sportId": 1, "season": season, "gameType": "R"})
            request = Request(
                f"{SCHEDULE_URL}?{params}",
                headers={"User-Agent": "baseball-prediction-model/1.0"},
            )
            with urlopen(request, timeout=90) as response:
                games = parse_schedule(json.load(response), season)
            minimum = 750 if season == 2020 else 2000
            if len(games) < minimum:
                raise ValueError(
                    f"Season {season} contains only {len(games)} completed games; "
                    f"expected at least {minimum}. Refusing an incomplete backtest."
                )
            print(f"Loaded {season}: {len(games)} completed regular-season games", flush=True)
            rows.extend(games)
        pd.DataFrame.from_records(rows).to_csv(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_games(
    path: Path,
    *,
    min_season: int = 2015,
    max_season: int = 2025,
) -> pd.DataFrame:
    """Return completed, non-tied regular-season games in date order."""
    games = pd.read_csv(path, low_memory=False)
    missing = REQUIRED_COLUMNS.difference(games.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    games = games.copy()
    if "playoff" in games.columns:  # Supports older manually supplied files.
        regular = games["playoff"].isna() | games["playoff"].astype(str).str.strip().eq("")
        games = games.loc[regular]
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    for column in ("season", "neutral", "score1", "score2"):
        games[column] = pd.to_numeric(games[column], errors="coerce")
    games = games.dropna(
        subset=["date", "season", "team1", "team2", "score1", "score2"]
    )
    games = games.loc[
        games["season"].between(min_season, max_season)
        & games["score1"].ne(games["score2"])
        & games["team1"].ne(games["team2"])
    ].copy()
    if "game_id" in games:
        games = games.drop_duplicates(subset=["game_id"])
    games["team1"] = games["team1"].astype(str)
    games["team2"] = games["team2"].astype(str)
    games["season"] = games["season"].astype(int)
    games["neutral"] = games["neutral"].fillna(0).astype(int)
    games["home_win"] = (games["score1"] > games["score2"]).astype(int)
    if games.empty:
        raise ValueError("No completed regular-season games in the requested years")
    return games.sort_values("date", kind="stable").reset_index(drop=True)
