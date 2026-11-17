"""Fit a simple NHL expected-goals (xG) model from PuckAPI shot locations.

    pip install -r requirements.txt
    export PUCKAPI_API_KEY=...        # free key: https://puckapi.com/signup
    python xg_model.py --season 20252026 --teams TOR EDM

Each page of shots costs 10 credits (up to 2,000 shots). One team's season is
about 3 pages, so two teams fit easily in the free 500 credits a month.
"""
import argparse
import math

import numpy as np
import pandas as pd
from puckapi import Client
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

GOAL_X = 89.0  # goal lines sit at x = +/-89 ft in the NHL rink frame


def fetch_shots(client, season, team):
    """Every shot attempt by one team in one season, following next_offset."""
    rows, offset = [], 0
    while offset is not None:
        page = client.get_shot_map(season=season, team=team, limit=2000, offset=offset)
        rows += page["shots"]
        offset = page.get("next_offset")
    return pd.DataFrame(rows)


def features(df):
    """Unblocked attempts (goals, shots on goal, misses) with distance and angle to the net."""
    df = df[df["type"].isin(["goal", "shot-on-goal", "missed-shot"])].dropna(subset=["x_coord", "y_coord"]).copy()
    # Shots are taken toward the nearer net; mirror everything onto the positive-x end.
    x = df["x_coord"].abs().astype(float)
    y = df["y_coord"].astype(float)
    df["distance"] = np.hypot(GOAL_X - x, y)
    df["angle"] = np.degrees(np.arctan2(y.abs(), (GOAL_X - x).clip(lower=0.1)))
    df["goal"] = (df["type"] == "goal").astype(int)
    X = pd.get_dummies(df[["distance", "angle", "shot_type", "strength"]], columns=["shot_type", "strength"], dtype=float)
    return X, df["goal"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="20252026", help="8-digit season id, e.g. 20252026")
    ap.add_argument("--teams", nargs="+", default=["TOR", "EDM"])
    args = ap.parse_args()

    client = Client()
    shots = pd.concat([fetch_shots(client, args.season, t) for t in args.teams], ignore_index=True)
    X, y = features(shots)
    print(f"{len(X):,} unblocked attempts, {y.sum():,} goals ({y.mean():.1%})")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)
    model = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    base = np.full_like(p, y_tr.mean())
    print(f"log loss {log_loss(y_te, p):.4f} vs {log_loss(y_te, base):.4f} for the league-average rate")
    print(f"AUC {roc_auc_score(y_te, p):.3f}")

    # xG by distance band, a quick calibration check.
    test = X_te.assign(goal=y_te.values, xg=p)
    bands = pd.cut(test["distance"], [0, 10, 20, 30, 45, 200])
    print(test.groupby(bands, observed=True)[["goal", "xg"]].mean().round(3).rename(columns={"goal": "actual"}))
    print("credits left:", client.last_credits_remaining)


if __name__ == "__main__":
    main()
