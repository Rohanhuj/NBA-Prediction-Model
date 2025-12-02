"""Minimal FastAPI service to serve NBA matchups with model predictions.

Endpoint: GET /api/matchups?date=YYYY-MM-DD

It pulls the daily schedule from balldontlie (no API key needed) and uses the
existing model assets under `model/` to generate a predicted winner and
confidence when historical data is available.

Run locally:
  uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

Notes:
- The model only covers seasons defined in simple_agent/src/agent/prediction_node
  (currently up to 2023-24 with cutoff 2024-07-01). Dates outside that range
  will return an error per game.
- balldontlie returns team abbreviations that align with the model data
  (e.g., DEN, LAL), which we pass directly into the predictor.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from functools import lru_cache
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import os

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "simple_agent" / "src"
if str(AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(AGENT_SRC))

# Reuse the model/prediction helpers from the LangGraph agent
from agent.prediction_node import (  # type: ignore
    _build_sample,
    _load_model,
    get_ranked_team_dict,
)

app = FastAPI(title="NBA Matchups API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"]
,
    allow_headers=["*"],
)


BALDONTLIE_BASE = "https://api.balldontlie.io"
BDL_API_KEY = os.getenv("BDL_API_KEY")


def _format_tipoff(dt_str: Optional[str]) -> str:
    if not dt_str:
        return "Tipoff TBA"
    try:
        # balldontlie returns ISO strings, often with Z suffix.
        parsed = dt.fromisoformat(dt_str.replace("Z", "+00:00"))
        return parsed.strftime("%-I:%M %p UTC").lstrip("0")
    except Exception:
        return dt_str


@lru_cache(maxsize=1)
def _client() -> httpx.AsyncClient:
    headers = {}
    if BDL_API_KEY:
        headers["Authorization"] = f"Bearer {BDL_API_KEY}"
    return httpx.AsyncClient(base_url=BALDONTLIE_BASE, timeout=10.0, headers=headers)


async def fetch_games_for_date(date: str) -> List[Dict[str, Any]]:
    client = _client()
    resp = await client.get(
        "/nba/v1/games",
        params={"dates[]": date, "per_page": 100},
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    payload = resp.json()
    return payload.get("data", [])


async def predict_game(home_abbr: str, away_abbr: str, game_date: str) -> Dict[str, Any]:
    try:
        team_dict, season_start, dict_file = get_ranked_team_dict(game_date)
    except Exception as exc:
        return {"error": str(exc)}

    model = _load_model()
    sample = _build_sample(team_dict, home_abbr, away_abbr, game_date, season_start)
    if sample is None:
        return {
            "error": "Not enough historical data for this matchup.",
            "season_file": dict_file,
            "end_date": season_start,
        }

    try:
        pred = await asyncio.to_thread(model.predict, sample)
        proba = await asyncio.to_thread(model.predict_proba, sample)
        label = int(pred[0])
        probs = proba[0].tolist()
        confidence = probs[label]
        winner = home_abbr if label == 1 else away_abbr
        return {
            "predictedWinner": winner,
            "confidence": float(confidence),
            "proba": [float(v) for v in probs],
            "season_file": dict_file,
            "end_date": season_start,
            "error": None,
        }
    except Exception as exc:
        return {"error": f"Prediction failed: {exc}"}


@app.get("/api/matchups")
async def get_matchups(date: str = Query(..., description="Date in YYYY-MM-DD")):
    """Return scheduled games for a date with model predictions when available."""
    try:
        dt.datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    games = await fetch_games_for_date(date)
    results = []

    for g in games:
        home = g.get("home_team", {})
        away = g.get("visitor_team", {})
        home_abbr = home.get("abbreviation")
        away_abbr = away.get("abbreviation")
        prediction = await predict_game(home_abbr, away_abbr, date) if home_abbr and away_abbr else {"error": "Missing team abbreviations."}

        results.append({
            "id": g.get("id"),
            "homeTeam": home.get("full_name") or home_abbr,
            "homeAbbr": home_abbr,
            "awayTeam": away.get("full_name") or away_abbr,
            "awayAbbr": away_abbr,
            "tipoff": _format_tipoff(g.get("datetime")),
            "gameDate": date,
            **prediction,
            # placeholders for future news synthesis
            "modelSummary": prediction.get("error") or "Model pick generated.",
            "newsPoints": [],
        })

    return {"date": date, "games": results}


@app.on_event("shutdown")
async def _shutdown_client():
    client = _client()
    await client.aclose()
