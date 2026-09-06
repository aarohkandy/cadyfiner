"""Override module for diagnose_codegen_reliability.py: routes generation+execution through
the self-repair candidate instead of the plain extract_code+run_cadquery path."""

from __future__ import annotations

from pathlib import Path

from cadyfiner.generators.local_ollama import generate as _ollama_generate
from cadyfiner.oracle.self_repair import generate_and_execute_with_repair


def generate_and_execute(prompt: str, out_dir: Path, generate_kwargs: dict):
    return generate_and_execute_with_repair(prompt, _ollama_generate, generate_kwargs, out_dir, max_repairs=1, run_cadquery_kwargs={"timeout_s": 90})
