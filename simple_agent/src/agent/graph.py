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
from tools import fetch_and_rank_news



@dataclass
class State:
    query: str
    chat_history: List[BaseMessage] = field(default_factory=list)
    agent_outcome : Union[AgentAction, AgentFinish, None] = None
    intermediate_steps: Annotated[List[tuple[AgentAction, str]], operator.add] = field(default_factory=list)


llm = Ollama(model="llama3")
tools = [fetch_and_rank_news]

agent_runnable = initialize_agent(
    tools=tools,
    llm=llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


async def run_agent(state: State) -> dict:
    result = await agent_runnable.ainvoke({
        "input": state.query,
        "chat_history": state.chat_history,
        })
    return {
        "query": result,
        "agent_outcome": None
    }


graph = (
    StateGraph(State)
    .add_node("run_agent", run_agent)
    .add_edge("__start__", "run_agent")
    .add_edge("run_agent", "__end__")
    .compile()
)
