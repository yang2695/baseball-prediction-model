"""Construct pregame features without looking at the outcome being predicted."""

from collections import deque
from dataclasses import dataclass, field

import pandas as pd


FEATURE_COLUMNS = [
    "home_recent_win_pct", "away_recent_win_pct",
    "home_recent_run_diff", "away_recent_run_diff",
    "home_recent_runs_scored", "away_recent_runs_scored",
    "home_recent_runs_allowed", "away_recent_runs_allowed",
    "home_season_win_pct", "away_season_win_pct",
    "home_season_run_diff", "away_season_run_diff",
    "home_games_played", "away_games_played",
    "home_rest_days", "away_rest_days",
    "neutral_site", "month",
]


@dataclass
class TeamHistory:
    wins: int = 0
    games: int = 0
    runs_for: int = 0
    runs_against: int = 0
    last_date: pd.Timestamp | None = None
    recent_wins: deque = field(default_factory=lambda: deque(maxlen=10))
    recent_scored: deque = field(default_factory=lambda: deque(maxlen=10))
    recent_allowed: deque = field(default_factory=lambda: deque(maxlen=10))

    def snapshot(self, date: pd.Timestamp) -> dict[str, float]:
        """Only games on earlier dates have been recorded at this point."""
        n = len(self.recent_wins)
        rest = 3 if self.last_date is None else min(
            7, max(0, (date - self.last_date).days - 1)
        )
        return {
            "recent_win_pct": sum(self.recent_wins) / n if n else 0.5,
            "recent_run_diff": (
                (sum(self.recent_scored) - sum(self.recent_allowed)) / n if n else 0.0
            ),
            "recent_runs_scored": sum(self.recent_scored) / n if n else 4.5,
            "recent_runs_allowed": sum(self.recent_allowed) / n if n else 4.5,
            "season_win_pct": self.wins / self.games if self.games else 0.5,
            "season_run_diff": (
                (self.runs_for - self.runs_against) / self.games
                if self.games else 0.0
            ),
            "games_played": float(self.games),
            "rest_days": float(rest),
        }

    def add_result(self, *, date: pd.Timestamp, scored: int, allowed: int) -> None:
        won = int(scored > allowed)
        self.wins += won
        self.games += 1
        self.runs_for += scored
        self.runs_against += allowed
        self.last_date = date
        self.recent_wins.append(won)
        self.recent_scored.append(scored)
        self.recent_allowed.append(allowed)


def make_features(games: pd.DataFrame) -> pd.DataFrame:
    """Build a game-level table in date order, updating teams AFTER each date.

    Grouping by date prevents doubleheader outcomes from becoming features for
    another game played the same day when first-pitch times are unavailable.
    Team histories reset every season; no current-game scores enter predictors.
    """
    required = {"date", "season", "team1", "team2", "score1", "score2", "home_win",
                "neutral", "elo_prob1"}
    if missing := required.difference(games.columns):
        raise ValueError(f"Missing columns for feature generation: {sorted(missing)}")
    if games.empty:
        raise ValueError("No games to featurize")

    ordered = games.sort_values("date", kind="stable")
    histories: dict[tuple[int, str], TeamHistory] = {}
    rows = []

    for date, day in ordered.groupby("date", sort=False):
        # Capture all pregame snapshots before recording ANY result from this date.
        for game in day.itertuples(index=False):
            home = histories.setdefault((int(game.season), game.team1), TeamHistory())
            away = histories.setdefault((int(game.season), game.team2), TeamHistory())
            home_snapshot = home.snapshot(date)
            away_snapshot = away.snapshot(date)
            features = {f"home_{key}": value for key, value in home_snapshot.items()}
            features.update(
                {f"away_{key}": value for key, value in away_snapshot.items()}
            )
            features["neutral_site"] = int(game.neutral)
            features["month"] = int(date.month)
            features.update(
                date=date,
                season=int(game.season),
                home_team=game.team1,
                away_team=game.team2,
                home_win=int(game.home_win),
                elo_prob_home=float(game.elo_prob1),
            )
            rows.append(features)
        for game in day.itertuples(index=False):
            histories[(int(game.season), game.team1)].add_result(
                date=date, scored=int(game.score1), allowed=int(game.score2)
            )
            histories[(int(game.season), game.team2)].add_result(
                date=date, scored=int(game.score2), allowed=int(game.score1)
            )
    return pd.DataFrame.from_records(rows)
