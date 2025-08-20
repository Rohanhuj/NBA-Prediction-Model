"""Define a data enrichment agent.

Works with a chat model with tool calling support.
"""

import json
from typing import Any, Dict, List, Literal, Optional, cast, Union, Annotated

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents import initialize_agent, AgentType
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_ollama import OllamaLLM as Ollama
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
import operator
from langgraph.graph import StateGraph
from agent.tools import fetch_and_rank_news
from agent.prediction_node import prediction_node
from agent.state import State


llm = Ollama(model="llama3")
tools = [fetch_and_rank_news]

agent_runnable = initialize_agent(
    tools=tools,
    llm=llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


async def run_agent(state: State) -> dict:
    predicted = state.predicted_winner
    opponent = state.away_team if predicted == state.home_team else state.home_team
    payload = {
        "query": state.query,
        "home_team": state.home_team,
        "away_team": state.away_team,
        "game_date": state.game_date,
        "predicted_winner": predicted,
        "opponent": opponent,
        "confidence": state.confidence,
    }
    result = await agent_runnable.ainvoke({
        "input": f"fetch_and_rank_news({json.dumps(payload)})",
        "chat_history": state.chat_history,
    })
    return {
        "query": state.query,
        "agent_outcome": result,
    }


graph = (
    StateGraph(State)
    .add_node("predict", prediction_node)
    .add_node("run_agent", run_agent)
    .add_edge("__start__", "predict")
    .add_edge("predict", "run_agent")
    .add_edge("run_agent", "__end__")
    .compile()
)
