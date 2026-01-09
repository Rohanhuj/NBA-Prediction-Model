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
from pydantic import BaseModel
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
AGENT_SRC = ROOT / "simple_agent" / "src"
if str(AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(AGENT_SRC))

# Reuse the model/prediction helpers from the LangGraph agent
from agent.prediction_node import (  # type: ignore
    _build_sample,
    _load_model,
    get_ranked_team_dict,
)
from agent.graph import graph  # type: ignore
from langchain_core.agents import AgentFinish

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
PREFETCH_DAYS = int(os.getenv("PREFETCH_DAYS", "3"))
PREFETCH_DELAY_SECONDS = float(os.getenv("PREFETCH_DELAY_SECONDS", "0.2"))
_matchups_cache: Dict[str, List[Dict[str, Any]]] = {}
_cache_lock = asyncio.Lock()


def api_date_to_graph_date(date: str) -> str:
    """Convert API YYYY-MM-DD date to graph/model MM-DD-YYYY format."""
    parsed = dt.datetime.strptime(date, "%Y-%m-%d")
    return parsed.strftime("%m-%d-%Y")


def _parse_api_date(date_str: str) -> dt.date:
    return dt.datetime.strptime(date_str, "%Y-%m-%d").date()


def _format_api_date(date_obj: dt.date) -> str:
    return date_obj.strftime("%Y-%m-%d")


class PredictRequest(BaseModel):
    game_id: int
    date: str
    query: Optional[str] = None


def _format_tipoff(dt_str: Optional[str]) -> str:
    if not dt_str:
        return "Tipoff TBA"
    try:
        # balldontlie returns ISO strings, often with Z suffix.
        parsed = dt.fromisoformat(dt_str.replace("Z", "+00:00"))
        date_part = parsed.strftime("%b %d, %Y").replace(" 0", " ")
        time_part = parsed.strftime("%I:%M %p").lstrip("0")
        return f"{date_part} • {time_part} UTC"
    except Exception:
        return dt_str


def _build_matchup(game: Dict[str, Any], game_date: str) -> Dict[str, Any]:
    home = game.get("home_team", {})
    away = game.get("visitor_team", {})
    home_abbr = home.get("abbreviation")
    away_abbr = away.get("abbreviation")
    return {
        "id": game.get("id"),
        "home_team": home.get("full_name") or home_abbr,
        "home_abbr": home_abbr,
        "away_team": away.get("full_name") or away_abbr,
        "away_abbr": away_abbr,
        "tipoff": _format_tipoff(game.get("datetime")),
        "game_date": game_date,
    }


@lru_cache(maxsize=1)
def _client() -> httpx.AsyncClient:
    headers = {}
    if BDL_API_KEY:
        headers["Authorization"] = f"Bearer {BDL_API_KEY}"
    return httpx.AsyncClient(base_url=BALDONTLIE_BASE, timeout=10.0, headers=headers)


async def fetch_games_for_date(date: str) -> List[Dict[str, Any]]:
    client = _client()
    try:
        resp = await client.get(
            "/nba/v1/games",
            params={"dates[]": date, "per_page": 100},
        )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Upstream request failed: {exc}") from exc
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    payload = resp.json()
    return payload.get("data", [])


def _prefetch_dates(center: dt.date, days: int = PREFETCH_DAYS) -> List[str]:
    return [
        _format_api_date(center + dt.timedelta(days=offset))
        for offset in range(-days, days + 1)
    ]


async def _ensure_cached_dates(date_str: str) -> None:
    target_date = _parse_api_date(date_str)
    dates = _prefetch_dates(target_date)
    async with _cache_lock:
        missing = [d for d in dates if d not in _matchups_cache]
    if not missing:
        return

    results: Dict[str, List[Dict[str, Any]]] = {}
    for date_key in missing:
        results[date_key] = await fetch_games_for_date(date_key)
        if PREFETCH_DELAY_SECONDS:
            await asyncio.sleep(PREFETCH_DELAY_SECONDS)
    async with _cache_lock:
        for date_key, games in results.items():
            _matchups_cache[date_key] = [_build_matchup(game, date_key) for game in games]


async def predict_game(home_abbr: str, away_abbr: str, game_date: str) -> Dict[str, Any]:
    graph_date = api_date_to_graph_date(game_date)
    try:
        team_dict, season_start, dict_file = get_ranked_team_dict(graph_date)
    except Exception as exc:
        return {"error": str(exc)}

    model = _load_model()
    sample = _build_sample(team_dict, home_abbr, away_abbr, graph_date, season_start)
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
            "predicted_winner": winner,
            "confidence": float(confidence),
            "proba": [float(v) for v in probs],
            "season_file": dict_file,
            "end_date": season_start,
            "error": None,
        }
    except Exception as exc:
        return {"error": f"Prediction failed: {exc}"}


def _extract_news_preview(agent_outcome: Any) -> Optional[str]:
    if agent_outcome is None:
        return None
    if isinstance(agent_outcome, AgentFinish):
        return agent_outcome.return_values.get("output") or str(agent_outcome)
    if isinstance(agent_outcome, dict):
        return agent_outcome.get("output") or agent_outcome.get("content") or str(agent_outcome)
    if hasattr(agent_outcome, "content"):
        return getattr(agent_outcome, "content")
    return str(agent_outcome)


@app.get("/api/matchups")
async def get_matchups(date: str = Query(..., description="Date in YYYY-MM-DD")):
    """Return scheduled games for a date with model predictions when available."""
    try:
        _parse_api_date(date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    await _ensure_cached_dates(date)
    async with _cache_lock:
        games = list(_matchups_cache.get(date, []))

    return {"date": date, "games": games}


@app.post("/api/predict")
async def predict_matchup(payload: PredictRequest):
    try:
        _parse_api_date(payload.date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    await _ensure_cached_dates(payload.date)
    async with _cache_lock:
        games = list(_matchups_cache.get(payload.date, []))
    game = next((g for g in games if g.get("id") == payload.game_id), None)
    if not game:
        raise HTTPException(status_code=404, detail="game_id not found for date")

    home_abbr = game.get("home_abbr")
    away_abbr = game.get("away_abbr")
    if not (home_abbr and away_abbr):
        raise HTTPException(status_code=400, detail="Missing team abbreviations for game.")

    model_prediction = await predict_game(home_abbr, away_abbr, payload.date)
    if model_prediction.get("error") or not model_prediction.get("predicted_winner"):
        return {
            "predicted_winner": None,
            "confidence": None,
            "news_preview_text": None,
            "error": model_prediction.get("error") or "Model prediction unavailable.",
            "news_status": "skipped_no_model_pick",
        }

    query = payload.query or f"{away_abbr} @ {home_abbr} NBA {payload.date} prediction"
    result = await graph.ainvoke({
        "query": query,
        "home_team": home_abbr,
        "away_team": away_abbr,
        "game_date": payload.date,
    })

    news_preview_text = result.get("news_preview_text") or _extract_news_preview(result.get("agent_outcome"))
    return {
        "predicted_winner": model_prediction.get("predicted_winner"),
        "confidence": model_prediction.get("confidence"),
        "news_preview_text": news_preview_text,
        "error": result.get("error"),
    }


@app.on_event("shutdown")
async def _shutdown_client():
    client = _client()
    await client.aclose()
