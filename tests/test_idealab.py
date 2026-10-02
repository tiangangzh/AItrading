"""Tests for aitrading.idealab (idea -> spec -> backtest -> interpretation -> files), fully offline.

Translators / interpreters: the offline heuristic ones and Claude-shaped ones driven by a
ScriptedLLM. The runner works on a small synthetic market with an injected official-factor loader.
"""

from __future__ import annotations

import json
import re
from datetime import date

import numpy as np
import pandas as pd
import pytest

from aitrading.backtest.metrics import compound
from aitrading.backtest.models import BacktestInterpretation, BacktestResult, CitedMetric
from aitrading.backtest.regression import FACTOR_COLUMNS
from aitrading.backtest.runner import StrategyRunner
from aitrading.data.synthetic import SyntheticProvider
from aitrading.discovery.models import ReplicationReport
from aitrading.idealab import NO_LLM, IdeaLab, IdeaLabResult
from aitrading.llm.base import ScriptedLLM
from aitrading.screen.spec import UniverseSpec
from aitrading.strategy.interpret import BacktestInterpreter, HeuristicInterpreter
from aitrading.strategy.library import TEMPLATES
from aitrading.strategy.nl import HeuristicStrategyTranslator, StrategyTranslator
from aitrading.strategy.spec import StrategySpec

END = date(2024, 12, 31)
TODAY = date(2026, 10, 2)
OPEN_UNIVERSE = UniverseSpec(min_price=None, min_avg_dollar_volume_usd_mn=None)


@pytest.fixture(scope="module")
def provider() -> SyntheticProvider:
    return SyntheticProvider(n_tickers=50, seed=5, start=date(2018, 1, 2), end=END)


def fake_french(provider: SyntheticProvider):
    def load(model: str, frequency: str) -> pd.DataFrame:
        bench = provider.get_benchmark_history(date(2018, 1, 2), END)
        r = (bench / bench.shift(1) - 1.0).dropna()
        rf = 0.0001
        if frequency == "monthly":
            r, rf = compound(r, "M"), 0.002
        rng = np.random.default_rng(7)
        n = len(r)
        df = pd.DataFrame({"Mkt-RF": r.to_numpy() - rf, "SMB": rng.normal(0, 0.02, n), "HML": rng.normal(0, 0.02, n),
                           "RMW": rng.normal(0, 0.01, n), "CMA": rng.normal(0, 0.01, n), "Mom": rng.normal(0, 0.03, n),
                           "RF": rf}, index=r.index)
        out = df[FACTOR_COLUMNS[model] + ["RF"]].copy()
        out.attrs["source"] = "Fake French"
        return out

    return load


@pytest.fixture(scope="module")
def runner(provider) -> StrategyRunner:
    return StrategyRunner(provider, factor_loader=fake_french(provider), default_years=4)


def _spec(key: str = "momentum_12_1", **update) -> StrategySpec:
    spec = TEMPLATES[key].spec()
    return spec.model_copy(update={"universe": OPEN_UNIVERSE, "start": date(2021, 1, 1), "end": END, **update})


# ------------------------------------------------------------------------------------------------
# Heuristic translator + heuristic interpreter
# ------------------------------------------------------------------------------------------------


def test_heuristic_idea_to_files(tmp_path, provider, runner):
    lab = IdeaLab(provider, translator=HeuristicStrategyTranslator(today=TODAY), runner=runner, out_dir=tmp_path)
    out = lab.run("12-1 momentum, long the top quintile, short the bottom quintile, from 2021")
    assert isinstance(out, IdeaLabResult)
    assert out.translation is not None and out.translation.template == "momentum_12_1"
    assert out.spec.name == "momentum_12_1" and out.spec.portfolio.n_quantiles == 5
    res = out.result
    assert res.interpretation is not None and out.interpretation is res.interpretation
    assert res.llm == NO_LLM and res.llm_calls == []
    assert all(c.status == "verified" for c in out.checks)  # the heuristic cites real numbers
    # files: <out_dir>/backtests/<run_id>/result.json + report.html
    assert out.json_path == tmp_path / "backtests" / res.run_id / "result.json"
    assert out.html_path == tmp_path / "backtests" / res.run_id / "report.html"
    saved = BacktestResult.model_validate_json(out.json_path.read_text(encoding="utf-8"))
    assert saved.run_id == res.run_id and saved.interpretation is not None
    assert saved.interpretation.verdict == res.interpretation.verdict
    page = out.html_path.read_text(encoding="utf-8")
    assert page.startswith("<!DOCTYPE html>")
    assert TEMPLATES["momentum_12_1"].title in page  # the matched template's title / description
    assert "SURVIVORSHIP BIAS" in page
    # the verdict respects the survivorship cap
    assert res.interpretation.verdict != "robust"


def test_run_with_spec_and_without_interpretation(tmp_path, provider, runner):
    lab = IdeaLab(provider, runner=runner, out_dir=tmp_path / "out")
    spec = _spec("low_volatility")
    out = lab.run(spec=spec, interpret=False)
    assert out.translation is None and out.spec is spec
    assert out.result.interpretation is None and out.checks == []
    assert out.json_path.exists() and out.html_path.exists()
    with pytest.raises(ValueError, match="give an idea"):
        lab.run()
    with pytest.raises(ValueError):
        lab.run("   ")


def test_out_dir_none_writes_nothing_and_backtests_folder_is_not_doubled(tmp_path, provider, runner):
    lab = IdeaLab(provider, runner=runner, out_dir=None)
    out = lab.run(spec=_spec(start=date(2023, 1, 1)))
    assert out.html_path is None and out.json_path is None and lab.runs_dir is None
    lab2 = IdeaLab(provider, runner=runner, out_dir=tmp_path / "backtests")
    out2 = lab2.run(spec=_spec(start=date(2023, 1, 1)))
    assert out2.json_path == tmp_path / "backtests" / out2.result.run_id / "result.json"


def test_runs_on_differently_configured_providers_do_not_overwrite_each_other(tmp_path):
    """Regression: the run id hashed only the provider name, so the same spec on SyntheticProvider seed 7 and seed 8
    shared a run folder and the second run silently replaced the first run's result.json / report.html."""
    spec = _spec(start=date(2023, 1, 1))
    outs = []
    for seed in (7, 8):
        p = SyntheticProvider(n_tickers=30, seed=seed, start=date(2021, 1, 4), end=END)
        lab = IdeaLab(p, runner=StrategyRunner(p, factor_loader=fake_french(p)), out_dir=tmp_path)
        outs.append(lab.run(spec=spec, interpret=False))
    a, b = outs
    assert a.result.run_id != b.result.run_id and a.json_path != b.json_path
    assert a.json_path.exists() and b.json_path.exists()
    saved_a = BacktestResult.model_validate_json(a.json_path.read_text(encoding="utf-8"))
    assert saved_a.stats["strategy"].cagr_pct == a.result.stats["strategy"].cagr_pct
    # a rerun of the same spec on the same provider configuration keeps its folder (deterministic id)
    p7 = SyntheticProvider(n_tickers=30, seed=7, start=date(2021, 1, 4), end=END)
    again = IdeaLab(p7, runner=StrategyRunner(p7, factor_loader=fake_french(p7)), out_dir=None).run(spec=spec, interpret=False)
    assert again.result.run_id == a.result.run_id


def test_default_components_are_offline(provider):
    lab = IdeaLab(provider, out_dir=None)
    assert isinstance(lab.translator, HeuristicStrategyTranslator)
    assert isinstance(lab.interpreter, HeuristicInterpreter)
    assert isinstance(lab.runner, StrategyRunner) and lab.runner.provider is provider


# ------------------------------------------------------------------------------------------------
# Claude-shaped translator / interpreter (ScriptedLLM)
# ------------------------------------------------------------------------------------------------


def _payload(user: str) -> dict:
    m = re.search(r"<result>\n(.*?)\n</result>", user, flags=re.S)
    assert m, "the interpreter prompt carries the result payload"
    return json.loads(m.group(1))


def _scripted(spec: StrategySpec, *, interpret: bool = True) -> ScriptedLLM:
    def translate(purpose, system, user, model):
        assert "<idea>" in user
        return spec

    def review(purpose, system, user, model):
        payload = _payload(user)
        sharpe = payload["stats"]["strategy"]["sharpe"]
        return BacktestInterpretation(
            summary="Momentum shows a modest spread on synthetic data.",
            verdict="robust",
            key_findings=[f"Sharpe {sharpe:.2f}."],
            cited_metrics=[CitedMetric(path="stats.strategy.sharpe", value=round(sharpe, 2), meaning="Sharpe")],
            biases_and_caveats=["Synthetic data."],
            next_experiments=["Try deciles."],
        )

    responders = {"strategy_spec": translate}
    if interpret:
        responders["interpret"] = review
    return ScriptedLLM(responders)


def test_scripted_llm_translator_and_interpreter(tmp_path, provider, runner):
    spec = _spec()
    llm = _scripted(spec)
    lab = IdeaLab(provider, translator=StrategyTranslator(llm, today=TODAY), interpreter=BacktestInterpreter(llm),
                  runner=runner, out_dir=tmp_path)
    out = lab.run("12-1 momentum quintiles since 2021")
    res = out.result
    assert out.translation is not None and out.translation.translator == "scripted"
    assert res.idea == "12-1 momentum quintiles since 2021"  # the translator keeps the verbatim idea
    assert res.llm == "scripted"
    # one shared LLM object: each call recorded once, in order
    assert [c.purpose for c in res.llm_calls] == ["strategy_spec", "interpret"]
    assert [c.status for c in out.checks] == ["verified"]
    # the reviewer said "robust" but the survivorship cap binds
    assert res.interpretation.verdict == "promising"
    assert "capped" in res.interpretation.summary.lower()
    saved = json.loads(out.json_path.read_text(encoding="utf-8"))
    assert [c["purpose"] for c in saved["llm_calls"]] == ["strategy_spec", "interpret"]
    # a second run only records its own calls
    out2 = lab.run(spec=spec)
    assert [c.purpose for c in out2.result.llm_calls] == ["interpret"]


def test_failed_llm_interpretation_falls_back_to_the_heuristic(tmp_path, provider, runner):
    spec = _spec(start=date(2022, 1, 1))
    llm = _scripted(spec, interpret=False)  # no responder for "interpret" -> LLMError
    lab = IdeaLab(provider, translator=StrategyTranslator(llm, today=TODAY), interpreter=BacktestInterpreter(llm),
                  runner=runner, out_dir=tmp_path)
    out = lab.run("momentum")
    res = out.result
    assert res.interpretation is not None
    assert any("LLM interpretation failed" in w for w in res.warnings)
    assert [c.purpose for c in res.llm_calls] == ["strategy_spec", "interpret"]
    assert res.llm_calls[-1].error == "no responder"
    assert res.llm == "scripted"


def test_factor_model_idea_end_to_end(tmp_path, provider, runner):
    lab = IdeaLab(provider, translator=HeuristicStrategyTranslator(today=TODAY), runner=runner, out_dir=tmp_path)
    out = lab.run("Fama-French 3 factor model")
    res = out.result
    assert out.spec.kind == "factor_model" and {"Mkt-RF", "SMB", "HML"} <= set(res.returns)
    assert [c.factor for c in res.factor_checks] == ["Mkt-RF", "SMB", "HML"]
    page = out.html_path.read_text(encoding="utf-8")
    assert TEMPLATES["ff3"].title in page and "Mkt-RF" in page


# ------------------------------------------------------------------------------------------------
# Replication
# ------------------------------------------------------------------------------------------------


def test_replicate_idea_writes_the_report_with_the_replication_section(tmp_path, provider, runner):
    lab = IdeaLab(provider, runner=runner, out_dir=tmp_path)
    lines: list[str] = []
    spec = _spec(start=date(2022, 1, 1))
    result, report, html_path = lab.replicate_idea(None, spec, progress=lines.append, min_years_per_split=1.0,
                                                   cost_levels_bps=(25.0,))
    assert isinstance(result, BacktestResult) and isinstance(report, ReplicationReport)
    assert report.backtest_run_id == result.run_id
    names = {c.name for c in report.checks}
    assert {"first_half", "second_half", "costs_25bps"} <= names
    assert any(c.passed is not None for c in report.checks if c.name in ("first_half", "second_half"))
    assert result.interpretation is not None
    folder = tmp_path / "backtests" / result.run_id
    assert html_path == folder / "report.html"
    page = html_path.read_text(encoding="utf-8")
    assert "Replication and robustness" in page and page.rstrip().endswith("</html>")
    assert page.index("Replication and robustness") < page.index('<footer class="disclaimer">')
    rep = ReplicationReport.model_validate_json((folder / "replication.json").read_text(encoding="utf-8"))
    assert rep.verdict == report.verdict
    assert BacktestResult.model_validate_json((folder / "result.json").read_text(encoding="utf-8")).run_id == result.run_id
    assert any("Replication verdict" in x for x in lines)
