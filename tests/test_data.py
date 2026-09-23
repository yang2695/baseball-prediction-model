"""Check MLB API normalization and basic data quality gates."""

import io

import pandas as pd
import pytest

from baseball_prediction.data import download_games, load_games, parse_schedule


def test_load_games_keeps_only_completed_regular_season_games(tmp_path):
    path = tmp_path / "games.csv"
    pd.DataFrame([
        dict(date="2022-04-07", season=2022, playoff="", team1="119",
             team2="137", neutral=0, score1=5, score2=3),
        dict(date="2022-10-12", season=2022, playoff="d", team1="119",
             team2="137", neutral=0, score1=6, score2=1),
        dict(date="2022-04-08", season=2022, playoff="", team1="119",
             team2="137", neutral=0, score1=None, score2=None),
        dict(date="2022-04-09", season=2022, playoff="", team1="119",
             team2="137", neutral=0, score1=2, score2=2),
        dict(date="2026-04-09", season=2026, playoff="", team1="119",
             team2="137", neutral=0, score1=2, score2=4),
    ]).to_csv(path, index=False)

    games = load_games(path)
    assert len(games) == 1
    assert games.iloc[0]["home_win"] == 1
    assert games.iloc[0]["season"] == 2022


def test_schedule_parser_ignores_unfinished_games():
    def make_game(pk, state, score):
        home = {"team": {"id": 119}}
        away = {"team": {"id": 137}}
        if score is not None:
            home["score"], away["score"] = score
        return {
            "gamePk": pk, "gameType": "R", "officialDate": "2025-04-01",
            "status": {"abstractGameState": state},
            "teams": {"home": home, "away": away},
        }

    payload = {"dates": [{"date": "2025-04-01", "games": [
        make_game(10, "Final", (4, 2)),
        make_game(11, "Preview", None),
        make_game(12, "Final", None),
        make_game(10, "Final", (4, 2)),
    ]}]}
    rows = parse_schedule(payload, 2025)
    assert len(rows) == 1
    assert rows[0]["score1"] == 4
    assert rows[0]["team1"] == "119"


def test_incomplete_api_result_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "baseball_prediction.data.urlopen",
        lambda request, timeout: io.BytesIO(b'{"dates": []}'),
    )
    destination = tmp_path / "raw.csv"
    with pytest.raises(ValueError, match="incomplete backtest"):
        download_games(destination, start_season=2025, end_season=2025)
    assert not destination.exists()
    assert not (tmp_path / "raw.csv.part").exists()


def test_missing_essential_column_fails_loudly(tmp_path):
    path = tmp_path / "broken.csv"
    pd.DataFrame({"date": ["2025-01-01"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing expected columns"):
        load_games(path)
