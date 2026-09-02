"""Generic LoRA supervised fine-tuning for both cadyfiner specialized models.

Both the Stage-2 gap-filler and the depth-policy mutation-proposer are
trained by this same script, parameterized by base model and data file —
they differ only in scale (1.5B vs 0.5B, matched to task complexity, see
docs/TRAINED_OPTIMIZERS.md) and in the shape of their JSON output, which
this script never needs to know about: training data is always
{"input": "<exact prompt text the frozen call site will send>", "output":
<a JSON-serializable object>} JSONL, and the model is trained to produce
`json.dumps(output)` as its completion for that input, verbatim.

CPU-only (this ran on homebase, which the project's CLAUDE.md documents as
having no GPU and a shared-tenant RAM/CPU budget — LoRA specifically
because full fine-tuning of even a 1.5B model's ~1.5B params on CPU is
impractical, while LoRA trains only the adapter, typically 0.1-1% of that).
"""

from __future__ import annotations

import argparse
import json

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def load_examples(path: str) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append({
                "messages": [
                    {"role": "user", "content": row["input"]},
                    {"role": "assistant", "content": json.dumps(row["output"], ensure_ascii=False)},
                ]
            })
    return examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True, help="HF model id, e.g. Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", required=True, help="JSONL of {input, output} rows")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-len", type=int, default=768)
    ap.add_argument("--max-steps", type=int, default=-1, help="Override for a quick smoke run")
    ap.add_argument("--eval-holdout-frac", type=float, default=0.1)
    args = ap.parse_args()

    examples = load_examples(args.data)
    if len(examples) < 4:
        raise SystemExit(f"only {len(examples)} training examples in {args.data} — too few to train on")

    n_eval = max(1, int(len(examples) * args.eval_holdout_frac))
    train_examples, eval_examples = examples[n_eval:], examples[:n_eval]
    print(f"train={len(train_examples)} eval={len(eval_examples)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="auto")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = Dataset.from_list(train_examples)
    eval_ds = Dataset.from_list(eval_examples)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        use_cpu=True,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_seq_len,
        logging_steps=1,
        eval_strategy="epoch" if args.max_steps == -1 else "no",
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        packing=False,
        dataset_text_field=None,  # using chat-formatted `messages`, not a flat text field
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds if args.max_steps == -1 else None,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
