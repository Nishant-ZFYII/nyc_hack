"""
agent/test_agent.py — Standalone test of the nat ReAct agent.

Run: python agent/test_agent.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Import to register our tool groups with the nat registry
import agent.register  # noqa: F401

from nat.runtime.loader import load_workflow
from nat.builder.workflow_builder import WorkflowBuilder


async def run(query: str) -> str:
    config_path = str(ROOT / "agent" / "config.yml")
    async with load_workflow(config_path) as workflow:
        async with workflow.run(query) as runner:
            # Try common result accessors across nat versions
            if hasattr(runner, "result"):
                r = runner.result
                result = await r() if callable(r) else r
            elif hasattr(runner, "get_result"):
                result = await runner.get_result()
            else:
                result = runner
            return str(result)


async def main():
    queries = [
        "I need a shelter tonight in Brooklyn near Flatbush",
        "I don't have an ID. Can I still apply for SNAP?",
        "What benefits qualify a family of 5 with $28K income?",
    ]
    for q in queries:
        print(f"\n{'=' * 72}")
        print(f"QUERY: {q}")
        print("=" * 72)
        try:
            result = await run(q)
            print(result)
        except Exception as e:
            print(f"ERROR: {e}")


if __name__ == "__main__":
    asyncio.run(main())
