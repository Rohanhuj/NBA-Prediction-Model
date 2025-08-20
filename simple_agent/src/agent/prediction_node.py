import pickle
import datetime
import numpy as np
import pandas as pd
from functools import lru_cache
import asyncio
from pathlib import Path
from typing import Optional, Tuple
import sys
import json


ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))


from generate_model import (
    encode, process_dict, find_recent_games, get_games_against_opponent,
    convert_MP, min_max_scaler, power_rank, standings_rank
)


MODEL_PKL = ROOT / "model" / "model.pkl"
DATA_DIR = ROOT / "model" / "data"

SEASONS = [
    ("10-26-2015", "07-01-2016", "dict15-16.pkl", "powerrankings15-16.pkl", "standings"),
    ("10-24-2016", "07-01-2017", "dict16-17.pkl", "powerrankings16-17.pkl", "standings"),
    ("10-16-2017", "07-01-2018", "dict17-18.pkl", "powerrankings17-18.json", "power", "04-11-2018"),
    ("10-15-2018", "07-01-2019", "dict18-19.pkl", "powerrankings18-19.json", "power", "04-10-2019"),
    ("12-21-2020", "07-30-2021", "dict20-21.pkl", "powerrankings20-21.json", "power", "05-16-2021"),
    ("10-18-2021", "07-01-2022", "dict21-22.pkl", "powerrankings21-22.json", "power", "04-10-2022"),
    ("10-24-2023", "07-01-2024", "dict23-24.pkl", "powerrankings23-24.json", "power", "07-01-2024"),
]

DATE_FMT = "%m-%d-%Y"

def _to_date(s: str) -> datetime.date:
    return datetime.datetime.strptime(s, DATE_FMT).date()



@lru_cache(maxsize=None)
def _load_team_dict(dict_filename: str):
    with open(DATA_DIR / dict_filename, "rb") as f:
        return pickle.load(f)
    
@lru_cache(maxsize=None)
def _load_rank(rank_file: str):
    path = DATA_DIR / rank_file
    if rank_file.endswith(".json"):
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    else:
        with path.open("rb") as f:
            return pickle.load(f) 
        
def get_ranked_team_dict(game_date : str):
    gd = _to_date(game_date)
    for start_s, end_s, dict_fname, rank_fname, method, *cutoff in SEASONS:
        if _to_date(start_s) <= gd < _to_date(end_s):
            team_dict = _load_team_dict(dict_fname)
            rank = _load_rank(rank_fname)
            if method == "power":
                ranked = power_rank(rank, team_dict, cutoff[0])
            elif method == "standings":
                ranked = standings_rank(rank, team_dict)
            return ranked, start_s, dict_fname
    raise ValueError(f"No season mapping covers {game_date}")


@lru_cache(maxsize=1)
def _load_model():
    with open(MODEL_PKL, "rb") as f:
        return pickle.load(f)
    
def _build_sample(team_dict: dict, team1: str, team2: str,
                  game_date: str, season_start: str) -> Optional[np.ndarray]:
    gd = _to_date(game_date)
    ed = _to_date(season_start)

    _, _, safeplayers, safeteam = encode(team_dict)
    new_dict = process_dict(team_dict, safeplayers, safeteam)

    t1r = find_recent_games(new_dict, team1, gd, ed)
    t2r = find_recent_games(new_dict, team2, gd, ed)
    t1v2 = get_games_against_opponent(new_dict, team1, team2, gd, ed)
    t2v1 = get_games_against_opponent(new_dict, team2, team1, gd, ed)
    if t1r.empty or t2r.empty or t1v2.empty or t2v1.empty:
        return None
    
    pad = pd.DataFrame(0, index=range(3), columns=t1r.columns)
    df = pd.concat([t1v2, pad, t1r, pad, t2v1, pad, t2r, pad])

    df = df[df["Starters"] != "Team Totals"]
    df = df.drop(["Starters", "Home"], axis=1)
    df["MP"] = df["MP"].apply(convert_MP)
    df = df.fillna(0).astype(float)
    df = min_max_scaler.fit_transform(df)

    flat = []
    for row in df:
        flat.extend(row.flatten())
    return np.array(flat).reshape(1, -1)

async def prediction_node(state):
    if not (state.home_team and state.away_team and state.game_date):
        return {"error": "home_team, away_team, and game_date are required."}

    try:
        team_dict, season_start, dict_file = get_ranked_team_dict(state.game_date)
    except ValueError as e:
        return {"error": str(e)}

    model = _load_model()
    team1 = state.home_team
    team2 = state.away_team

    X = _build_sample(team_dict, team1, team2, state.game_date, season_start)
    if X is None:
        return {
            "end_date": season_start,
            "season_file": dict_file,
            "error": "Not enough historical data for this matchup.",
        }

    pred  = await asyncio.to_thread(model.predict, X)
    proba = await asyncio.to_thread(model.predict_proba, X)
    pred_label = int(pred[0])
    p = proba[0].tolist()
    confidence = p[pred_label]
    predicted_winner = team1 if pred_label == 1 else team2

    return {
        "end_date": season_start,
        "season_file": dict_file,
        "predicted_winner": predicted_winner,
        "confidence": float(confidence),
        "raw_pred_label": pred_label,
        "proba": [float(v) for v in p],
        "error": None,
    }
