import asyncio
from graph import graph, State


async def run():
    payload = {
        "query": "Lakers vs Warriors NBA 2019-04-04 Prediction",
        "home_team": "LAL",
        "away_team": "GSW",
        "game_date": "2019-04-04",
    }
    result = await graph.ainvoke(payload)
    print("== MODEL PICK ==")
    print("Predicted:", result.get("predicted_winner"))
    conf = result.get("confidence")
    if conf is not None:
        print("Confidence:", f"{conf:.1%}")
    print("\n== EVIDENCE-BACKED PREVIEW (Agent Output) ==\n")
    print(result.get("news_preview_text"))

if __name__ == "__main__":
    asyncio.run(run())
