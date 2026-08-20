"""Run the exit-criteria paired evaluation over the held-out seed bank splits."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadyfiner.generators.local_ollama import generate
from cadyfiner.harness import run_paired_evaluation, summarize, SeedCase
from cadyfiner.spec import DesignBrief

FAMILIES_DIR = Path(__file__).resolve().parents[1] / "prompts" / "seed_bank" / "families"
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "prompts" / "seed_bank" / "manifest.json"

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemma4:e4b"
ROLES = sys.argv[2].split(",") if len(sys.argv) > 2 else ["heldout_same_family", "heldout_family"]


def load_seeds() -> list[SeedCase]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    all_items = {}
    for path in FAMILIES_DIR.glob("*.json"):
        for item in json.loads(path.read_text()):
            all_items[item["id"]] = item

    seeds = []
    for role, block in manifest["split"].items():
        if role not in ROLES:
            continue
        for seed_id in block["seed_ids"]:
            item = all_items[seed_id]
            seeds.append(
                SeedCase(
                    id=item["id"],
                    family=item["family"],
                    raw_prompt=item["raw_prompt"],
                    ground_truth=DesignBrief.model_validate(item["ground_truth"]),
                    role=role,
                )
            )
    return seeds


def main() -> None:
    seeds = load_seeds()
    print(f"Loaded {len(seeds)} seeds for roles {ROLES}, model={MODEL}")

    results = run_paired_evaluation(
        seeds, generate, reps=1,
        out_root=Path("workspace/harness") / MODEL.replace(":", "_"),
        generate_kwargs={"model": MODEL, "temperature": 0.5, "max_tokens": 2500, "timeout": 240},
    )
    for r in results:
        outcome = "REFINED WINS" if r.refined_pass and not r.raw_pass else \
                  "RAW WINS" if r.raw_pass and not r.refined_pass else "TIE"
        print(f"  {r.seed_id:<32} raw={r.raw_pass!s:6} refined={r.refined_pass!s:6}  {outcome}")

    report = summarize(results)
    print(f"\n=== EXIT CRITERIA REPORT ({MODEL}) ===")
    print(f"n_pairs={report.n_pairs}  n_distinct_seeds={report.n_distinct_seeds}  n_excluded_fallback={report.n_excluded_fallback}")
    print(f"refined_wins={report.refined_wins}  raw_wins={report.raw_wins}  ties={report.ties}")
    print(f"win_rate={report.win_rate_excluding_ties:.1%}  95% CI={report.win_rate_ci95}  sign_test_p={report.sign_test_p:.4f}")
    print(f"\nBy family:")
    for fam, b in report.by_family.items():
        print(f"  {fam:<22} refined_wins={b['refined_wins']} raw_wins={b['raw_wins']} ties={b['ties']} n={b['n']}")
    print(f"\nBy role:")
    for role, b in report.by_role.items():
        print(f"  {role:<22} refined_wins={b['refined_wins']} raw_wins={b['raw_wins']} ties={b['ties']} n={b['n']}")
    print(f"\nVERDICT: {report.verdict}")

    out_path = Path("workspace/harness") / MODEL.replace(":", "_") / "report.json"
    out_path.write_text(json.dumps({
        "model": MODEL, "n_pairs": report.n_pairs, "n_distinct_seeds": report.n_distinct_seeds,
        "n_excluded_fallback": report.n_excluded_fallback, "refined_wins": report.refined_wins,
        "raw_wins": report.raw_wins, "ties": report.ties,
        "win_rate": report.win_rate_excluding_ties, "ci95": list(report.win_rate_ci95),
        "sign_test_p": report.sign_test_p, "by_family": report.by_family, "by_role": report.by_role,
        "verdict": report.verdict,
    }, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
