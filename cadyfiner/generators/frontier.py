"""Frontier-model generator adapter (Claude via the Anthropic API).

This is the "deploy" backend per the user's own generator split: local
Ollama for fast unattended iteration, a frontier model for the deployed
target. Requires ``ANTHROPIC_API_KEY`` in the environment — not set in the
development environment this was built in, so this module has not been
exercised end-to-end; it errors clearly rather than silently rather than
pretending to work. Run it once with a real key before trusting it.
"""

from __future__ import annotations

import os


class FrontierGenerationError(RuntimeError):
    pass


def generate(
    prompt: str,
    *,
    model: str = "claude-sonnet-4-5",
    temperature: float = 0.4,
    max_tokens: int = 1500,
) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise FrontierGenerationError(
            "ANTHROPIC_API_KEY is not set. This generator backend needs a real API key "
            "supplied by the user before it can run — export it and retry."
        )

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as exc:
        raise FrontierGenerationError(f"Anthropic API call failed: {exc}") from exc

    return "".join(block.text for block in response.content if hasattr(block, "text"))
