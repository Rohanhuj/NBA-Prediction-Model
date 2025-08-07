"""Tools for data enrichment.

This module contains functions that are directly exposed to the LLM as tools.
These tools can be used for tasks such as web searching and scraping.
Users can edit and extend these tools as needed.
"""

import json
from typing import Any, Optional, cast, List

import aiohttp
from langchain_community.tools.tavily_search import TavilySearchResults
from tavily import TavilyClient
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from langgraph.prebuilt import InjectedState
from typing_extensions import Annotated
from langchain_ollama import OllamaLLM as Ollama
import asyncio
from langchain_core.tools import tool
from enrichment_agent.configuration import Configuration
from enrichment_agent.state import State
from enrichment_agent.utils import init_model
from dotenv import load_dotenv
from newspaper import Article
import re
import os
load_dotenv()

tavily = TavilyClient(api_key = os.getenv("TAVILY_API_KEY"))
llm = Ollama(model = "llama3")
def fetch_game_news(query: str, num_results = 3):
    """Fetch news articles related to a specific game or topic."""
    
    try:
        # Use the Tavily search tool to get results based on the query
        results = tavily.search(query, max_results = num_results, search_depth = "advanced")
        return results["results"]
    except Exception as e:
        return [{"error fetching articles": str(e)}]

def scrape_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text.strip()
    except Exception as e:
        print(f"Error scraping article: {e}")
        return ""
    
async def scrape_article_text_async(url: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, scrape_article_text, url)

def is_box_score_url(url: str) -> bool:
    keywords = ["boxscore", "recap", "gameId", "final-score", "/game/", "game-summary"]
    return any(k in url.lower() for k in keywords)


async def async_rank_articles(query: str, summaries: list[str]) -> str:
    summary_list = "\n\n".join([f"[{i}] {s}" for i, s in enumerate(summaries)])
    rank_prompt = (
        f"Rank the following article summaries in order of relevance to predicting the outcome "
        f"of the game: '{query}'. Return the top 3 indices and explain your choice:\n\n"
        f"{summary_list}\n\nTop 3 indices and rationale:"
    )
    return await llm.ainvoke(rank_prompt)

async def async_summarize_article(title, source, text):
    summary_prompt = (
        f"Summarize the following article to extract player availability, team momentum, betting odds, "
        f"injuries, analyst picks, and overall sentiment about who you would predict to win:\n\n"
        f"Title: {title}\nSource: {source}\n\nText:\n{text[:5000]}\n\nSummary:"
    )
    return await llm.ainvoke(summary_prompt)

async def fetch_articles_async(query: str, num_results: int = 10) -> List[dict]:
    search_query = f"{query} game preview prediction odds injuries -boxscore -recap"
    return await asyncio.to_thread(tavily.search, search_query, max_results=num_results, search_depth="advanced", exclude_domains=["espn.com", "nba.com"])

@tool("fetch_and_rank_news", return_direct = True)
async def fetch_and_rank_news(query: str) -> str:
    """Fetch and rank news articles related to a specific game or topic."""
    response = await fetch_articles_async(query)
    raw_articles = response.get("results", [])
    if not raw_articles:
        return "No articles found."
    articles = []
    summaries = []
    for article in raw_articles:
        url = article.get("url")
        print(url)
        if not url or is_box_score_url(url):
            "skipping box score or invalid URL"
            continue

        text = await scrape_article_text_async(url)
        title = article.get("title", "Untitled")
        source = article.get("source", {}).get("name", "Unknown")

        try:
            summary = await async_summarize_article(title, source, text)
        except Exception as e:
            summary = f"Error summarizing: {e}"

        summaries.append(summary)
        articles.append({
            "title": title,
            "source": source,
            "url": url,
            "summary": summary,
        })

    try:
        ranking = await async_rank_articles(query, summaries)
        top_indices = list(map(int, re.findall(r"\[(\d+)\]", ranking)))[:3]
    except Exception as e:
        ranking = f"Error ranking articles: {e}"

    top_articles = [articles[i] for i in top_indices if i < len(articles)]

    formatted_articles = "\n\n---\n\n".join([
        f"Title: {a['title']}\nSource: {a['source']}\nURL: {a['url']}\n\nSummary: {a['summary']}"
        for a in top_articles
    ])

    synthesis_prompt = (
        f"Based on the following article summaries, compile a single unified report "
        f"about the upcoming game: '{query}'.\n\n"
        f"Include:\n"
        f"- Player availability (without assuming final outcomes)\n"
        f"- Team momentum\n"
        f"- Analyst picks\n"
        f"- Betting odds\n"
        f"- General sentiment\n\n"
        f"**Exclude**:\n"
        f"- Any final scores or box scores\n"
        f"- Game results — assume the game has not happened yet.\n\n"
        f"Summaries:\n\n"
        f"{formatted_articles}\n\n"
        f"Unified Game Preview:"
)

    print(synthesis_prompt)
    try:
        final_summary = await llm.ainvoke(synthesis_prompt)
    except Exception as e:
        final_summary = f"Error generating unified summary: {e}"

    return (
        f"Ranking for '{query}':\n\n{ranking}\n\n"
        f"Unified Game Preview:\n\n{final_summary}"
    )


    #return f"Ranking for '{query}':\n\n{ranking}\n\nSummaries:\n\n{formatted_articles}"


    