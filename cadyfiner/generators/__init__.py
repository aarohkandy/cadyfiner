"""Generator adapters: raw prompt text in, raw LLM output text out.

Every adapter exposes the same signature —
``generate(prompt: str, *, temperature: float, max_tokens: int) -> str`` —
so :mod:`cadyfiner.optimize` and :mod:`cadyfiner.harness` can swap generators
without caring which one is underneath. Callers run the returned text
through :func:`cadyfiner.oracle.execute.extract_code` before executing it.
"""
