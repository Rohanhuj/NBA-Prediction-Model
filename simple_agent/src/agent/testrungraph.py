import asyncio
from graph import graph, State


async def run():
    input_state = State(
        query="Lakers vs Celtics NBA 01-23-2025",
        chat_history=[],
        intermediate_steps=[],
    )
    result = await graph.ainvoke(input_state)
    print("\n✅ FINAL OUTPUT:")
    print(result)

if __name__ == "__main__":
    asyncio.run(run())
