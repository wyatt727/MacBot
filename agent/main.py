# agent/main.py
import asyncio
from .agent import MinimalAIAgent
from .config import LLM_MODEL

async def main():
    agent = MinimalAIAgent(model=LLM_MODEL)
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())
