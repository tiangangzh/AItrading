"""Train an open-weights "judge" model on your belief ledger with GRPO (reinforcement learning).

    aitrading beliefs export judge.jsonl --test-from 2023-01-01 --llm-cutoff 2024-06-01
    python scripts/train_judge_grpo.py judge.jsonl --baseline-only          # what the Bayesian engine scores
    pip install "trl>=0.19" datasets peft accelerate                        # GPU machine
    python scripts/train_judge_grpo.py judge.jsonl --model Qwen/Qwen2.5-1.5B-Instruct --lora --out judge-ckpt
    # serve it (e.g. `vllm serve judge-ckpt`) and use it:
    aitrading beliefs judge "5-day momentum" --judge open --judge-url http://localhost:8000/v1 --judge-model judge-ckpt

Each example is one hypothesis with the experience log as it was BEFORE its result was known
(``aitrading.beliefs.dataset.build_examples``). The model answers with a probability that the
strategy's Sharpe ratio is positive; the reward is the log score of that probability for what
actually happened, relative to a coin flip (``judge_reward``) - a proper scoring rule, so the
policy is paid for calibration, not for optimism.

Honest expectations - read before spending GPU hours:
* The reward is noisy: one year of returns pins a Sharpe ratio down to about +/-1. You need
  hundreds to thousands of resolved hypotheses (ideally out-of-sample ones) before a trained judge
  can beat the Bayesian engine; ``--baseline-only`` shows the score to beat on the test split.
* Outcomes in one period are correlated (a regime lifts every momentum variant at once): split by
  time (``--test-from`` at export), never at random.
* The base model has read papers about the backtest period. Prefer targets after its knowledge
  cutoff (``after_llm_cutoff`` in the export) for evaluation.

This script is a starting point. It is not exercised by the test suite (it needs a GPU and the
optional ``trl`` / ``datasets`` packages); the dataset, prompt format and reward it uses are.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _load(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        sys.exit(f"{path} has no examples (record backtests with learning on, then: aitrading beliefs export FILE)")
    return rows


def baseline(rows: list[dict]) -> dict[str, float]:
    """Mean reward of the Bayesian engine's probability and of a coin flip, per split."""
    from aitrading.beliefs.dataset import judge_reward

    out: dict[str, float] = {}
    for split in ("train", "test"):
        part = [r for r in rows if r.get("split") == split]
        if not part:
            continue
        bayes = [judge_reward(json.dumps({"p_positive": r["bayes_p_positive"]}), r["outcome_positive"]) for r in part]
        base_rate = sum(r["outcome_positive"] for r in part) / len(part)
        const = [judge_reward(json.dumps({"p_positive": base_rate}), r["outcome_positive"]) for r in part]
        out[f"{split}_n"] = len(part)
        out[f"{split}_bayes_reward"] = sum(bayes) / len(bayes)
        out[f"{split}_base_rate_reward"] = sum(const) / len(const)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path, help="JSONL from `aitrading beliefs export`")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="base model (Hugging Face id or path)")
    ap.add_argument("--out", default="judge-ckpt", help="output directory")
    ap.add_argument("--baseline-only", action="store_true", help="only print the scores a trained judge must beat")
    ap.add_argument("--oos-only", action="store_true", help="train only on out-of-sample targets (holdout, paper)")
    ap.add_argument("--lora", action="store_true", help="train LoRA adapters instead of all weights (needs peft)")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--generations", type=int, default=8, help="completions sampled per prompt (GRPO group size)")
    ap.add_argument("--batch", type=int, default=8, help="per-device batch size (a multiple of --generations)")
    args = ap.parse_args(argv)

    rows = _load(args.data)
    scores = baseline(rows)
    print(json.dumps(scores, indent=2))
    if args.baseline_only:
        return 0
    train_rows = [r for r in rows if r.get("split") != "test" and (r.get("out_of_sample") or not args.oos_only)]
    if len(train_rows) < 50:
        print(f"warning: only {len(train_rows)} training examples; a trained judge will mostly learn noise", file=sys.stderr)

    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        sys.exit('training needs: pip install "trl>=0.19" datasets accelerate (and peft for --lora), on a GPU machine')

    from aitrading.beliefs.dataset import judge_reward

    train = Dataset.from_list([{"prompt": r["messages"], "outcome_positive": bool(r["outcome_positive"])} for r in train_rows])

    def reward(completions, outcome_positive, **_):  # noqa: ANN001 - TRL calls it with dataset columns as kwargs
        out = []
        for c, y in zip(completions, outcome_positive):
            text = c[-1]["content"] if isinstance(c, list) else str(c)
            out.append(judge_reward(text, bool(y)))
        return out

    peft_config = None
    if args.lora:
        from peft import LoraConfig

        peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    config = GRPOConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        num_generations=args.generations,
        max_completion_length=256,
        temperature=1.0,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )
    trainer = GRPOTrainer(model=args.model, reward_funcs=reward, args=config, train_dataset=train, peft_config=peft_config)
    trainer.train()
    trainer.save_model(args.out)
    print(f"saved to {args.out}. Score it on the test split by serving it and running "
          f"`aitrading beliefs calibration` after new holdout / paper evidence arrives, or compare its mean reward on the "
          f"test rows with test_bayes_reward = {scores.get('test_bayes_reward', math.nan):.3f}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
