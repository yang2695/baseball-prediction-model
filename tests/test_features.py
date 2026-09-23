"""Pregame information must never include the game being predicted."""

import pandas as pd
import pytest

from baseball_prediction.features import FEATURE_COLUMNS, make_features


def game(date, season, home, away, home_score, away_score):
    return dict(date=pd.Timestamp(date), season=season, team1=home, team2=away,
                score1=home_score, score2=away_score,
                home_win=int(home_score > away_score), neutral=0, elo_prob1=0.5)


def test_same_date_games_do_not_leak_into_one_another():
    games = pd.DataFrame([
        game("2021-04-01", 2021, "H", "A", 10, 1),
        game("2021-04-01", 2021, "H", "B", 0, 2),
        game("2021-04-02", 2021, "H", "C", 4, 1),
    ])
    result = make_features(games)

    assert result.iloc[0]["home_recent_win_pct"] == 0.5
    assert result.iloc[1]["home_recent_win_pct"] == 0.5
    assert result.iloc[0]["home_games_played"] == 0
    assert result.iloc[1]["home_games_played"] == 0
    assert result.iloc[2]["home_games_played"] == 2
    assert result.iloc[2]["home_recent_win_pct"] == 0.5
    assert result.iloc[2]["home_recent_runs_scored"] == 5.0
    assert result.iloc[2]["home_rest_days"] == 0
    assert not {"score1", "score2", "home_win", "elo_prob_home"}.intersection(
        FEATURE_COLUMNS
    )


def test_history_resets_between_seasons_and_tracks_rest():
    games = pd.DataFrame([
        game("2021-04-01", 2021, "H", "A", 9, 1),
        game("2021-04-05", 2021, "H", "B", 3, 2),
        game("2022-04-01", 2022, "H", "A", 1, 2),
    ])
    result = make_features(games)

    assert result.iloc[1]["home_rest_days"] == 3
    assert result.iloc[1]["home_season_win_pct"] == 1.0
    assert result.iloc[2]["home_games_played"] == 0
    assert result.iloc[2]["home_season_win_pct"] == 0.5


def test_rejects_missing_feature_input():
    with pytest.raises(ValueError, match="Missing columns"):
        make_features(pd.DataFrame({"date": [pd.Timestamp("2022-04-01")]}))
