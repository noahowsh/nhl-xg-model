# nhl-xg-model

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/noahowsh/nhl-xg-model/blob/main/xg_model.ipynb)

A small, readable NHL expected-goals (xG) model you can run in a minute: pull a season of shot locations, build distance and angle features, fit a logistic regression and check its calibration.

```bash
pip install -r requirements.txt
export PUCKAPI_API_KEY=your_key      # free: https://puckapi.com/signup (500 credits a month)
python xg_model.py --season 20252026 --teams TOR EDM
```

Example output (one team, 2025-26):

```
3,274 unblocked attempts, 252 goals (7.7%)
log loss 0.2440 vs 0.2712 for the league-average rate
AUC 0.736
```

## How it works

- **Data:** `get_shot_map` from [PuckAPI](https://puckapi.com/docs?utm_source=github&utm_medium=readme&utm_campaign=nhl-xg-model) returns every shot attempt with rink coordinates, shot type, strength, shooter and goalie, 2,000 per page (10 credits). One team's season is about three pages.
- **Features:** distance and angle to the net (every shot mirrored onto one end), shot type, and strength (5v5, PP, PK). Blocked shots are dropped, so this is an unblocked-attempt (Fenwick) model.
- **Model:** scikit-learn logistic regression, a 25% holdout, log loss against the league-average rate, AUC, and actual vs predicted goal rate by distance band.

## Ideas to extend it

- Add rebounds and rush shots, using the time and distance since the previous event (`get_plays`).
- Train on more teams or seasons; play-by-play goes back to 2010-11.
- Score goalies on goals saved above expected.
- The same tools work for the AHL, PWHL, OHL, WHL, QMJHL and ECHL: pass `league="ahl"`.

Uses the [puckapi](https://pypi.org/project/puckapi/) Python client (no dependencies). PuckAPI is also a hosted MCP server, if you'd rather ask Claude or ChatGPT: https://puckapi.com/mcp?utm_source=github&utm_medium=readme&utm_campaign=nhl-xg-model

MIT licensed.
