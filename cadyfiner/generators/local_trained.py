"""Specialized locally-trained model adapter.

Loads a base model + a LoRA adapter fine-tuned specifically for one
cadyfiner task (see training/train_lora.py and docs/TRAINED_OPTIMIZERS.md)
and exposes it through the same ``generate(prompt, **kwargs) -> str``
contract every other adapter in this package uses, so it's a drop-in
alternative to :mod:`cadyfiner.generators.local_ollama` — swap which
`generate` callable `refine_stage2.fill_gaps` or `optimize.py`'s mutation
proposer is given, nothing else about the calling code changes.

This does NOT replace the general-purpose Ollama/frontier path — both are
kept. This adapter is for the two narrow, specialized tasks a small
fine-tuned model was trained for; it will not be a good general CAD-code
generator, and isn't meant to be.
"""

from __future__ import annotations

from functools import lru_cache

_DEFAULT_MAX_NEW_TOKENS = 512


class TrainedModelError(RuntimeError):
    pass


@lru_cache(maxsize=4)
def _load(base_model: str, adapter_path: str):
    """Load (and cache) a base model + LoRA adapter pair.

    Cached by (base_model, adapter_path) so repeated ``generate()`` calls
    in the same process — the normal case, e.g. one optimizer run or one
    harness sweep — pay the load cost once, not per call.
    """

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise TrainedModelError(
            "local_trained requires torch/transformers/peft — install with "
            "`pip install -e '.[trained]'`"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto")
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer, torch


def generate(
    prompt: str,
    *,
    base_model: str,
    adapter_path: str,
    temperature: float = 0.3,
    max_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
    **_ignored,
) -> str:
    """Run one generation through a specialized fine-tuned model.

    ``base_model``/``adapter_path`` are required kwargs (no sensible
    default — a caller must say which specialized model it wants), unlike
    the other adapters' ``model`` default, so a caller can never
    accidentally fall through to a wrong or missing adapter silently.
    """

    model, tokenizer, torch = _load(base_model, adapter_path)

    # Template to text, then tokenize separately (rather than
    # apply_chat_template(..., return_tensors="pt") directly): that call
    # returns a BatchEncoding in this transformers version, not a raw
    # tensor, and passing a BatchEncoding as model.generate()'s positional
    # `input_ids` arg fails with a confusing AttributeError deep in
    # generate() rather than at the call site. Splitting the two steps
    # keeps the return type explicit and unpacked with `**`, which is
    # correct for either a BatchEncoding or a bare-tensor return.
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-4),
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = output_ids[0][encoded["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
