"""Checks for dataset filtering and schema validation."""

import pandas as pd
import pytest

from baseball_prediction.data import load_games


def test_load_games_keeps_only_finished_regular_season_games(tmp_path):
    path = tmp_path / "games.csv"
    pd.DataFrame([
        dict(date="2022-04-07", season=2022, playoff="", team1="LAD",
             team2="SFG", neutral=0, score1=5, score2=3, elo_prob1=0.62),
        dict(date="2022-10-12", season=2022, playoff="d", team1="LAD",
             team2="SFG", neutral=0, score1=6, score2=1, elo_prob1=0.70),
        dict(date="2022-04-08", season=2022, playoff="", team1="LAD",
             team2="SFG", neutral=0, score1=None, score2=None, elo_prob1=0.60),
        dict(date="2022-04-09", season=2022, playoff="", team1="LAD",
             team2="SFG", neutral=0, score1=2, score2=2, elo_prob1=0.60),
        dict(date="2023-04-09", season=2023, playoff="", team1="LAD",
             team2="SFG", neutral=0, score1=2, score2=4, elo_prob1=0.55),
    ]).to_csv(path, index=False)

    games = load_games(path)
    assert len(games) == 1
    assert games.iloc[0]["home_win"] == 1
    assert games.iloc[0]["season"] == 2022


def test_missing_essential_column_fails_loudly(tmp_path):
    path = tmp_path / "broken.csv"
    pd.DataFrame({"date": ["2022-01-01"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing expected columns"):
        load_games(path)
