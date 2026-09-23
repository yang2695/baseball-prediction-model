"""Evaluation is held out by season, not a random game-level split."""

import numpy as np
import pandas as pd
import pytest

from baseball_prediction.features import FEATURE_COLUMNS
from baseball_prediction.model import score_probabilities, train_and_evaluate


def sample_features():
    records = []
    for season in range(2015, 2026):
        for index in range(12):
            record = {feature: float((index + season) % 7) for feature in FEATURE_COLUMNS}
            record.update(
                date=pd.Timestamp(season, 4, 1) + pd.Timedelta(days=index),
                season=season,
                home_team="LAD",
                away_team="SFG",
                home_win=index % 2,
                elo_prob_home=0.52,
            )
            records.append(record)
    return pd.DataFrame(records)


def test_training_keeps_2025_for_final_test():
    model, report, predictions = train_and_evaluate(sample_features())

    assert report["game_counts"] == {
        "initial_training": 108, "validation": 12, "final_training": 120, "test": 12
    }
    assert predictions["season"].eq(2025).all()
    assert predictions["predicted_home_win_probability"].between(0, 1).all()
    assert report["selected_model"] in report["validation"]
    assert set(report["test"]) == {
        "selected_model", "historical_home_rate", "simple_pregame_elo"
    }
    assert model is not None


def test_probability_scoring_rewards_well_calibrated_predictions():
    good = score_probabilities([1, 0], np.array([0.9, 0.1]))
    bad = score_probabilities([1, 0], np.array([0.1, 0.9]))
    assert good["log_loss"] < bad["log_loss"]
    assert good["brier_score"] < bad["brier_score"]


def test_missing_test_season_is_not_silently_evaluated():
    frame = sample_features()
    with pytest.raises(ValueError, match="Need 2015"):
        train_and_evaluate(frame.loc[frame["season"] < 2025])
