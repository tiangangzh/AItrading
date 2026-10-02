"""Representative task, end to end: the canonical demo observation -> ScreenSpec -> screen ->
ranked candidates -> explained, grounded dislocations -> Markdown report.

Offline by default: synthetic market data, the heuristic translator and the heuristic explainer
(no API key; everything except the run timestamps is deterministic). ``--live`` uses Claude
(``AnthropicLLM``) to translate the observation and explain the candidates; it needs
ANTHROPIC_API_KEY (or another Anthropic credential).

    python examples/representative_task.py               # offline
    python examples/representative_task.py --live        # Claude
    python examples/representative_task.py --save-run runs

The report is written to examples/output/representative_task.md (``--report`` to change it) and a
short summary is printed. On synthetic data the summary also shows the archetype the provider
planted for each explained name, so the explanation can be checked against ground truth.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from aitrading.agent.explain import Explainer
from aitrading.agent.offline import HeuristicExplainer
from aitrading.cli import credentials_help, is_auth_error
from aitrading.core.models import PipelineResult
from aitrading.data.synthetic import SyntheticProvider
from aitrading.pipeline import ResearchPipeline
from aitrading.report.markdown import render_markdown
from aitrading.screen.nl import HeuristicScreenTranslator, NLScreenTranslator
from aitrading.screen.spec import ScreenSpec

OBSERVATION = (
    "Find US mid-caps ($2-20B) that were in established uptrends (50-day above 200-day, positive 12-1 momentum) but "
    "have pulled back 15-40% from their 52-week highs over the past few months on heavy volume, now oversold (RSI "
    "under 40), still generating strong free cash flow (FCF yield above 4%) with revenue growth above 8%, and where "
    "short interest is elevated (above 6% of float). Rank by FCF yield, growth and the size of the drawdown, then "
    "read the latest earnings calls and explain the dislocation."
)
AS_OF = date(2026, 9, 30)
CANONICAL_CONDITIONS = [
    "market_cap_usd_bn between 2 and 20",
    "sma_50_vs_sma_200_pct > 0",
    "return_12m_ex_1m_pct > 0",
    "drawdown_from_52w_high_pct between -40 and -15",
    "max_volume_ratio_20d >= 2",
    "rsi_14 < 40",
    "fcf_yield_pct > 4",
    "revenue_growth_yoy_pct > 8",
    "short_interest_pct_float > 6",
]
CANONICAL_RANKING = [
    ("fcf_yield_pct", "higher_is_better"),
    ("revenue_growth_yoy_pct", "higher_is_better"),
    ("drawdown_from_52w_high_pct", "lower_is_better"),
]
DEFAULT_REPORT = Path(__file__).resolve().parent / "output" / "representative_task.md"


def build_pipeline(live: bool, *, model: str | None, effort: str, explain_top_k: int, out_dir: Path | None) -> ResearchPipeline:
    provider = SyntheticProvider()
    if live:
        from aitrading.llm.anthropic_client import AnthropicLLM

        llm = AnthropicLLM(model=model, effort=effort)
        translator, explainer = NLScreenTranslator(llm, effort=effort), Explainer(llm, effort=effort)
    else:
        translator, explainer = HeuristicScreenTranslator(), HeuristicExplainer()
    return ResearchPipeline(provider, translator, explainer, out_dir=out_dir, explain_top_k=explain_top_k)


def matches_canonical(spec: ScreenSpec) -> bool:
    """Same conditions (in any order) and the same ranking factors / directions as the canonical spec."""
    conditions = sorted(c.describe() for c in spec.all_conditions())
    ranking = sorted((f.feature, f.direction) for f in spec.ranking)
    return conditions == sorted(CANONICAL_CONDITIONS) and ranking == sorted(CANONICAL_RANKING)


def summarize(result: PipelineResult, provider: object, report_path: Path) -> str:
    spec = ScreenSpec.model_validate(result.spec)
    arrows = {"higher_is_better": "(higher)", "lower_is_better": "(lower)"}
    explained = [i for i in result.ideas if i.thesis is not None]
    lines = [
        f"Representative task as of {result.as_of} | provider {result.provider} | LLM {result.llm}",
        f"Spec: {len(spec.all_conditions())} conditions; ranking "
        + ", ".join(f"{f.feature} {arrows[f.direction]}" for f in spec.ranking)
        + f"; matches the canonical demo spec: {'yes' if matches_canonical(spec) else 'no'}",
        f"Funnel: {result.universe_size} names -> {result.survivors} passed -> top {len(result.ideas)} ranked "
        f"-> {len(explained)} explained",
    ]
    archetype = getattr(provider, "archetype", None)
    for idea in result.ideas:
        c = idea.candidate
        if idea.error:
            lines.append(f"  #{c.rank:<2} {c.ticker:<6} explanation failed: {idea.error[:120]}")
            continue
        if idea.thesis is None:
            continue
        t, g = idea.thesis, idea.grounding
        grounding = f"{g.n_verified}/{len(g.checks)}" if g is not None else "n/a"
        planted = ""
        if callable(archetype):
            try:
                planted = f"  [planted: {archetype(c.ticker)}]"
            except KeyError:
                planted = ""
        lines.append(
            f"  #{c.rank:<2} {c.ticker:<6} {c.name[:30]:<30} {t.dislocation_type:<32} {t.conviction:<6} "
            f"{'actionable' if t.is_actionable else 'not actionable':<14} grounded {grounding}{planted}"
        )
    if result.llm_calls:
        tokens_in = sum(c.input_tokens for c in result.llm_calls)
        tokens_out = sum(c.output_tokens for c in result.llm_calls)
        lines.append(f"LLM calls: {len(result.llm_calls)} ({tokens_in:,} input / {tokens_out:,} output tokens)")
    if result.warnings:
        lines.append(f"Warnings: {len(result.warnings)} (listed in the report's appendix)")
    lines.append(f"Report: {report_path}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the representative research task end to end.")
    parser.add_argument("--live", action="store_true", help="use Claude for translation and explanations")
    parser.add_argument("--model", default=None, help="Claude model id (live only)")
    parser.add_argument("--effort", default="high", choices=("low", "medium", "high", "xhigh", "max"), help="Claude effort (live only)")
    parser.add_argument("--explain", type=int, default=5, metavar="K", help="explain the top K candidates (default 5)")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="where to write the Markdown report")
    parser.add_argument("--save-run", type=Path, default=None, metavar="DIR", help="also write the run artifacts under DIR")
    args = parser.parse_args(argv)

    try:
        pipe = build_pipeline(args.live, model=args.model, effort=args.effort, explain_top_k=max(0, args.explain), out_dir=args.save_run)
        result = pipe.run(OBSERVATION, AS_OF)
    except Exception as exc:  # noqa: BLE001 - report a missing key helpfully, re-raise anything else
        if args.live and is_auth_error(exc):
            print(credentials_help(f"{type(exc).__name__}: {exc}"), file=sys.stderr)
            print("(In this example the offline mode is the default: run it without --live.)", file=sys.stderr)
            return 1
        raise
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_markdown(result), encoding="utf-8")
    print(summarize(result, pipe.provider, args.report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
