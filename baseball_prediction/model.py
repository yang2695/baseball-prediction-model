"""Train, select, and evaluate calibrated win-probability candidates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseball_prediction.features import FEATURE_COLUMNS


def candidate_models() -> dict[str, Any]:
    """Return fresh estimators. Neither one has access to final-score columns."""
    return {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, random_state=42),
        ),
        "gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.04,
                max_leaf_nodes=15,
                min_samples_leaf=40,
                l2_regularization=1.0,
                random_state=42,
            ),
        ),
    }


def score_probabilities(actual: pd.Series | np.ndarray, probabilities: np.ndarray) -> dict:
    """Use probability quality, not just winner-picking accuracy."""
    y = np.asarray(actual, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "games": int(len(y)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
    }


def train_and_evaluate(features: pd.DataFrame) -> tuple[Any, dict, pd.DataFrame]:
    """Select on 2021, refit through 2021, and touch 2022 only for final testing.

    Test labels never enter feature generation, imputation, scaling, model fitting,
    or model selection. FiveThirtyEight's Elo is a comparison only, not a feature.
    """
    if missing := set(FEATURE_COLUMNS + ["season", "home_win", "elo_prob_home",
                                           "date", "home_team", "away_team"]).difference(features):
        raise ValueError(f"Missing modeling columns: {sorted(missing)}")
    train = features.loc[features["season"].between(2015, 2020)].copy()
    validation = features.loc[features["season"].eq(2021)].copy()
    test = features.loc[features["season"].eq(2022)].copy()
    if any(part.empty for part in (train, validation, test)):
        counts = features["season"].value_counts().sort_index().to_dict()
        raise ValueError(f"Need 2015–2020 training, 2021 validation, and 2022 test games; available: {counts}")
    if train["home_win"].nunique() < 2:
        raise ValueError("Training data must contain both winners and losers")

    validation_scores = {}
    for name, model in candidate_models().items():
        model.fit(train[FEATURE_COLUMNS], train["home_win"])
        validation_scores[name] = score_probabilities(
            validation["home_win"],
            model.predict_proba(validation[FEATURE_COLUMNS])[:, 1],
        )
    selected_name = min(
        validation_scores,
        key=lambda name: validation_scores[name]["log_loss"],
    )

    # Train anew on all allowable years AFTER choosing the model family.
    development = pd.concat([train, validation], ignore_index=True)
    selected_model = candidate_models()[selected_name]
    selected_model.fit(development[FEATURE_COLUMNS], development["home_win"])

    probabilities = selected_model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    prior_home_win_rate = float(development["home_win"].mean())
    naive_probabilities = np.full(len(test), prior_home_win_rate)
    elo_probabilities = test["elo_prob_home"].to_numpy(dtype=float)
    summary = {
        "selection_metric": "2021 validation log loss (lower is better)",
        "selected_model": selected_name,
        "seasons": {"train": "2015–2020", "validation": 2021, "test": 2022},
        "game_counts": {
            "initial_training": int(len(train)),
            "validation": int(len(validation)),
            "final_training": int(len(development)),
            "test": int(len(test)),
        },
        "features": FEATURE_COLUMNS,
        "validation": validation_scores,
        "test": {
            "selected_model": score_probabilities(test["home_win"], probabilities),
            "historical_home_rate": score_probabilities(
                test["home_win"], naive_probabilities
            ),
            "fivethirtyeight_pregame_elo": score_probabilities(
                test["home_win"], elo_probabilities
            ),
        },
        "baseline_home_win_probability": prior_home_win_rate,
    }
    predictions = test[
        ["date", "season", "home_team", "away_team", "home_win", "elo_prob_home"]
    ].copy()
    predictions["predicted_home_win_probability"] = probabilities
    predictions["baseline_home_win_probability"] = prior_home_win_rate
    return selected_model, summary, predictions
