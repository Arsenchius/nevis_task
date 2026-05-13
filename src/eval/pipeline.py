from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

from src.eval.evaluation import validate_transcript_package
from src.eval.ground_truth import generate_ground_truth_for_spec
from src.eval.scenarios import DatasetSpec
from src.eval.transcripts import generate_transcript_for_package
from src.eval.utils import (
    save_ground_truth_package,
    save_transcript_package,
    save_validated_package,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - cosmetic fallback
    tqdm = None


async def run_stage_async(
    client: AsyncOpenAI,
    items: list[Any],
    worker: Callable[[AsyncOpenAI, Any], Awaitable[dict[str, Any]]],
    *,
    stage_name: str,
    max_concurrency: int = 4,
    save_fn=None,
    show_progress: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_one(item: Any) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
        item_id = getattr(item, "example_id", None) or item.get("example_id", "unknown")
        async with semaphore:
            try:
                package = await worker(client, item)
                if save_fn is not None:
                    await asyncio.to_thread(save_fn, package)
                return package, None
            except Exception as exc:
                return None, {
                    "example_id": item_id,
                    "stage": stage_name,
                    "error": repr(exc),
                }

    tasks = [asyncio.create_task(run_one(item)) for item in items]
    results = []

    progress = None
    if show_progress and tqdm is not None:
        progress = tqdm(
            total=len(tasks),
            desc=stage_name,
            unit="example",
            dynamic_ncols=True,
        )

    passed = 0
    failed = 0
    for task in asyncio.as_completed(tasks):
        package, failure = await task
        results.append((package, failure))
        if package is not None:
            passed += 1
        else:
            failed += 1
        if progress is not None:
            progress.set_postfix(passed=passed, failed=failed)
            progress.update(1)

    if progress is not None:
        progress.close()

    packages = [package for package, _ in results if package is not None]
    failures = [failure for _, failure in results if failure is not None]
    packages.sort(key=lambda item: item["example_id"])
    failures.sort(key=lambda item: item["example_id"])
    return packages, failures


async def generate_ground_truths_async(
    client: AsyncOpenAI,
    specs: list[DatasetSpec],
    *,
    max_concurrency: int = 4,
    save: bool = True,
    show_progress: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    return await run_stage_async(
        client,
        specs,
        generate_ground_truth_for_spec,
        stage_name="ground_truth",
        max_concurrency=max_concurrency,
        save_fn=save_ground_truth_package if save else None,
        show_progress=show_progress,
    )


async def generate_transcripts_async(
    client: AsyncOpenAI,
    ground_truth_packages: list[dict[str, Any]],
    *,
    max_concurrency: int = 4,
    save: bool = True,
    show_progress: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    return await run_stage_async(
        client,
        ground_truth_packages,
        generate_transcript_for_package,
        stage_name="transcript",
        max_concurrency=max_concurrency,
        save_fn=save_transcript_package if save else None,
        show_progress=show_progress,
    )


async def validate_examples_async(
    client: AsyncOpenAI,
    transcript_packages: list[dict[str, Any]],
    *,
    max_concurrency: int = 2,
    save: bool = True,
    show_progress: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    return await run_stage_async(
        client,
        transcript_packages,
        validate_transcript_package,
        stage_name="validation",
        max_concurrency=max_concurrency,
        save_fn=save_validated_package if save else None,
        show_progress=show_progress,
    )


async def generate_dataset_async(
    client: AsyncOpenAI,
    specs: list[DatasetSpec],
    *,
    generation_concurrency: int = 4,
    validation_concurrency: int = 2,
    save: bool = True,
    show_progress: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    ground_truth_packages, gt_failures = await generate_ground_truths_async(
        client,
        specs,
        max_concurrency=generation_concurrency,
        save=save,
        show_progress=show_progress,
    )
    if gt_failures:
        logger.warning(
            "%d/%d ground-truth specs failed; those examples will be skipped in transcript generation. "
            "Failed IDs: %s",
            len(gt_failures),
            len(specs),
            [f["example_id"] for f in gt_failures],
        )
    transcript_packages, transcript_failures = await generate_transcripts_async(
        client,
        ground_truth_packages,
        max_concurrency=generation_concurrency,
        save=save,
        show_progress=show_progress,
    )
    validated_packages, validation_failures = await validate_examples_async(
        client,
        transcript_packages,
        max_concurrency=validation_concurrency,
        save=save,
        show_progress=show_progress,
    )
    failures = gt_failures + transcript_failures + validation_failures
    failures.sort(key=lambda item: (item["stage"], item["example_id"]))
    return validated_packages, failures
