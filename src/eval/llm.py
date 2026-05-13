from __future__ import annotations

import asyncio
from typing import Any

from openai import AsyncOpenAI

from src.eval.config import MODEL_CONFIGS


async def generate_with_llm(
    client: AsyncOpenAI,
    task_name: str,
    system_prompt: str,
    user_prompt: str,
    *,
    max_retries: int = 3,
) -> str:
    """Call OpenAI asynchronously for a named generation stage and return raw text."""
    config = MODEL_CONFIGS[task_name]
    request: dict[str, Any] = {
        "model": config["model"],
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "timeout": config["timeout_seconds"],
    }

    if config.get("temperature") is not None:
        request["temperature"] = config["temperature"]
    if config.get("reasoning") is not None:
        request["reasoning"] = config["reasoning"]

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = await client.responses.create(**request)
            return response.output_text.strip()
        except Exception as exc:
            last_error = exc
            if attempt == max_retries - 1:
                break
            await asyncio.sleep(2**attempt)

    raise RuntimeError(
        f"{task_name} LLM call failed after {max_retries} attempts: {last_error!r}"
    ) from last_error
