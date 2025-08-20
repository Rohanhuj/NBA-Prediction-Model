from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class State:
    query: str
    chat_history: List[Any] = field(default_factory=list)

    # inputs
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    game_date: Optional[str] = None  # "MM-DD-YYYY"

    # derived by the node (season start it used)
    end_date: Optional[str] = None   # "MM-DD-YYYY"
    season_file: Optional[str] = None

    # outputs
    predicted_winner: Optional[str] = None
    confidence: Optional[float] = None
    raw_pred_label: Optional[int] = None
    proba: Optional[List[float]] = None
    error: Optional[str] = None

    news_explanation: Optional[str] = None
