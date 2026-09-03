"""Local Ollama generator adapter.

HTTP call pattern ported from cadybara-kitchen's
``cadybara/providers/ollama.py`` (the ``/api/generate`` request shape is
already validated in this exact project family) — simplified to a single
blocking call, since cadyfiner's optimizer/harness loops run unattended and
don't need the interruptible-streaming variant that exists there for a
human-watched CLI.
"""

from __future__ import annotations

import time

import httpx


class OllamaGenerationError(RuntimeError):
    pass


def generate(
    prompt: str,
    *,
    model: str = "qwen2.5-coder:7b",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.4,
    max_tokens: int = 1500,
    timeout: float = 300.0,
    seed: int | None = None,
    max_retries: int = 2,
    think: bool = False,
) -> str:
    """Blocking call to Ollama's /api/generate, with retry on transient failures.

    Retries matter here specifically because Ollama serializes requests to
    one loaded model — a slow generation already in flight (this project
    routinely sees 60-250s calls on CPU-only inference) can push a
    concurrent caller's request past its own timeout even though the
    server is healthy and would have served it a few seconds later. Found
    live: a harness run crashed entirely on the very first call, with zero
    seeds completed, because of exactly this contention with other work
    running concurrently against the same local Ollama server.

    ``think=False`` by default: found live that a reasoning-capable model
    (``gemma4:e4b``, which lists "thinking" in its capabilities) served
    through a newer Ollama build (0.32.6, on homebase) silently spends the
    entire ``num_predict`` budget on hidden reasoning tokens and returns an
    EMPTY ``response`` field with ``done_reason: "length"`` — no error, no
    warning, just nothing, unless reasoning is explicitly turned off. The
    same model through this project's local (older) Ollama app never
    triggered it, so this is a real behavioral difference between Ollama
    versions for the identical model file, not a one-off fluke — worth
    disabling explicitly everywhere rather than trusting a given server's
    default.
    """

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": think,
        "options": {"temperature": temperature, "num_predict": max_tokens, "seed": seed},
    }
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        started = time.perf_counter()
        try:
            response = httpx.post(f"{base_url.rstrip('/')}/api/generate", json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            return str(data.get("response", ""))
        except httpx.HTTPError as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(2 * (attempt + 1))
                continue
            raise OllamaGenerationError(
                f"Ollama request failed after {max_retries + 1} attempts, last one took "
                f"{time.perf_counter() - started:.1f}s: {last_error}. "
                f"Is `ollama serve` running and is `{model}` pulled? (If the server is healthy but busy "
                f"with a concurrent request, consider raising `timeout`.)"
            ) from last_error
    raise AssertionError("unreachable")  # loop always returns or raises
