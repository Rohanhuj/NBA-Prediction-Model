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
from dotenv import load_dotenv
from newspaper import Article
import re
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / "simple_agent" / ".env")
load_dotenv(ROOT / ".env")


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


async def async_rank_articles(query: str, summaries: list[str], articles: list[dict], predicted_winner : str, opponent: str) -> str:
    items_block = []
    for i, (s, a) in enumerate(zip(summaries, articles)):
        items_block.append(
            f"[{i}]\nTitle: {a.get('title', 'Untitled')}\nURL: {a.get('url','')}\nSummary:\n{s}"
        )
    summary_list = "\n\n".join(items_block)
    rank_prompt = (
        f"You are ranking news for explainability.\n\n"
        f"Game: {query}\n"
        f"Model pick: {predicted_winner} over {opponent}\n\n"
        "Rank the summaries by how well they SUPPORT this pick with concrete evidence:\n"
        "- injuries/availability favoring the pick\n"
        "- form/matchup trends favoring the pick\n"
        "- odds/market movement aligning with the pick\n"
        "- analyst/beat-writer lean for the pick\n\n"
        "Prefer more recent and more concrete sourcing.\n"
        "Return a Ranking of the JSON with the TOP THREE separate from the rest with a rating, website URL, and a reason why it was rated/ranked:\n\n"
        f"{summary_list}\n\nJSON:"
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
    return await asyncio.to_thread(tavily.search, search_query, max_results=num_results, search_depth="advanced")


@tool("fetch_and_rank_news", return_direct = True)
async def fetch_and_rank_news(payload : str) -> str:
    """Fetch and rank news articles related to a specific game or topic."""

    try:
        data = json.loads(payload)
    except Exception as e:
        return f"Invalid JSON payload: {e}"

    query = data.get("query") or ""
    predicted_winner = data.get("predicted_winner")
    opponent = data.get("opponent")
    confidence = data.get("confidence")

    if not (predicted_winner and opponent):
        return ("fetch_and_rank_news now expects a model pick.\n"
                "Provide predicted_winner and opponent to rank supportively "
                "and synthesize an explanation.")


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
        ranking = await async_rank_articles(query, summaries, articles, predicted_winner, opponent)
        top_indices = list(map(int, re.findall(r"\[(\d+)\]", ranking)))[:3]
    except Exception as e:
        ranking = f"Error ranking articles: {e}"

    
    top_articles = [articles[i] for i in top_indices if i < len(articles)]
    

    formatted_articles = "\n\n---\n\n".join([
        f"Title: {a['title']}\nSource: {a['source']}\nURL: {a['url']}\n\nSummary: {a['summary']}"
        for a in top_articles
    ])

    conf_txt = f"{confidence:.1%}" if isinstance(confidence, (int, float)) else "N/A"

    synthesis_prompt = (
        f"Create a concise, evidence-backed explanation for why **{predicted_winner}** over **{opponent}** "
        f"is a reasonable pick for: {query}. Model confidence: {conf_txt}.\n\n"
        "Use only the article summaries below as evidence. Attribute claims to their sources. Focus on:\n"
        "- injuries/availability favoring the pick\n"
        "- recent form & matchup trends favoring the pick\n"
        "- odds/market movement aligning with the pick\n"
        "- analyst/beat-writer lean in the same direction\n\n"
        "Avoid outcome spoilers or post-game recaps; treat as pre-game context.\n\n"
        f"Articles:\n\n{formatted_articles}\n\n"
        "Now produce:\n"
        "1) 3–6 bullet **Evidence summary** with source attributions\n"
        "2) A short **Why this supports the pick** paragraph tying the evidence to the winner\n"
    )


    try:
        final_summary = await llm.ainvoke(synthesis_prompt)
    except Exception as e:
        final_summary = f"Error generating unified summary: {e}"

    return (
        f"Ranking for '{query}':\n\n{ranking}\n\n"
        f"Summaries:\n\n{formatted_articles}\n\n"
        f"Unified Game Preview:\n\n{final_summary}"
    )


    #return f"Ranking for '{query}':\n\n{ranking}\n\nSummaries:\n\n{formatted_articles}"


    
