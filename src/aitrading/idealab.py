"""Idea lab: research idea -> StrategySpec -> point-in-time backtest -> verdict -> saved report.

``IdeaLab(provider).run("12-1 momentum deciles")`` chains the platform's pieces:

1. **Translate** the idea with a :class:`~aitrading.strategy.nl.StrategyTranslator` (Claude) or the
   offline :class:`~aitrading.strategy.nl.HeuristicStrategyTranslator` (default) - or take a ready
   ``StrategySpec``.
2. **Backtest** it with :class:`~aitrading.backtest.runner.StrategyRunner` (point-in-time features,
   execution lag, costs, survivorship and data-provenance warnings).
3. **Interpret** the result with :class:`~aitrading.strategy.interpret.BacktestInterpreter` (Claude, a
   skeptical reviewer whose every cited number is verified) or the offline
   :class:`~aitrading.strategy.interpret.HeuristicInterpreter` (default). If the LLM interpreter fails,
   the heuristic one is used and a warning says so. The interpretation is attached to the result.
4. **Save** ``<out_dir>/backtests/<run_id>/result.json`` (the full ``BacktestResult``) and
   ``report.html`` (``render_backtest_html``, with the template's title / description / references
   when ``spec.name`` is a library template key). When ``out_dir`` itself is a folder named
   ``backtests`` the run folders go directly below it (so ``.../backtests/backtests`` never
   appears). ``out_dir=None`` writes nothing. The run id hashes the spec, the provider's name and
   configuration, the universe, the window and the execution lag, so two different experiments
   never share a folder; re-running the very same one replaces its files.

LLM audit: the calls made by the translator's and the interpreter's ``llm`` objects during the run
are appended to ``result.llm_calls`` and ``result.llm`` names the models used
(``"none (offline heuristics)"`` when neither uses an LLM).

:meth:`IdeaLab.replicate_idea` runs the replication / robustness suite
(:func:`aitrading.discovery.replicate.replicate`) on a spec through the same runner, interprets the
base run, and saves ``result.json``, ``replication.json`` and ``report.html`` (the backtest report
with the replication section appended) in the base run's folder.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from aitrading.backtest.models import BacktestResult
from aitrading.backtest.runner import StrategyRunner
from aitrading.core.models import EvidenceCheck, LLMCallRecord
from aitrading.screen.catalog import FeatureCatalog, default_catalog
from aitrading.strategy.interpret import HeuristicInterpreter
from aitrading.strategy.library import TEMPLATES
from aitrading.strategy.nl import HeuristicStrategyTranslator, StrategyTranslation
from aitrading.strategy.spec import StrategySpec

__all__ = ["IdeaLab", "IdeaLabResult", "NO_LLM"]

NO_LLM = "none (offline heuristics)"


@dataclass
class IdeaLabResult:
    """What one idea-lab run produced."""

    spec: StrategySpec
    translation: StrategyTranslation | None
    result: BacktestResult
    checks: list[EvidenceCheck] = field(default_factory=list)  # verification of the interpretation's cited metrics
    html_path: Path | None = None
    json_path: Path | None = None

    @property
    def interpretation(self) -> Any:
        return self.result.interpretation


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class IdeaLab:
    """Idea -> spec -> backtest -> interpretation -> files (see the module docstring).

    Args:
        provider: the ``MarketDataProvider`` to backtest on (ignored when ``runner`` is given).
        translator: object with ``translate(idea) -> StrategyTranslation`` (default: heuristic).
        interpreter: object with ``interpret(result) -> (BacktestInterpretation, checks)`` (default: heuristic).
        runner: a :class:`~aitrading.backtest.protocols.BacktestRunner` (default: ``StrategyRunner(provider)``).
        out_dir: where run folders are written (``None`` = do not write files).
        catalog: feature catalog for the default translator / runner.
    """

    def __init__(
        self,
        provider: Any,
        *,
        translator: Any | None = None,
        interpreter: Any | None = None,
        runner: Any | None = None,
        out_dir: Path | str | None = "aitrading_output",
        catalog: FeatureCatalog | None = None,
    ) -> None:
        self.provider = provider
        self.catalog = catalog or default_catalog()
        self.translator = translator if translator is not None else HeuristicStrategyTranslator(self.catalog)
        self.interpreter = interpreter if interpreter is not None else HeuristicInterpreter()
        self.runner = runner if runner is not None else StrategyRunner(provider, catalog=self.catalog)
        self.out_dir = Path(out_dir) if out_dir is not None else None

    # ------------------------------------------------------------------ paths
    @property
    def runs_dir(self) -> Path | None:
        """Folder holding one sub-folder per run (``<out_dir>/backtests``)."""
        if self.out_dir is None:
            return None
        return self.out_dir if self.out_dir.name == "backtests" else self.out_dir / "backtests"

    def run_dir(self, run_id: str) -> Path | None:
        base = self.runs_dir
        return None if base is None else base / run_id

    # ------------------------------------------------------------------ LLM bookkeeping
    def _llms(self) -> list[Any]:
        out: list[Any] = []
        for owner in (self.translator, self.interpreter):
            llm = getattr(owner, "llm", None)
            if llm is not None and isinstance(getattr(llm, "calls", None), list) and all(llm is not o for o in out):
                out.append(llm)
        return out

    def _marks(self) -> list[tuple[Any, int]]:
        return [(llm, len(llm.calls)) for llm in self._llms()]

    @staticmethod
    def _new_calls(marks: list[tuple[Any, int]]) -> list[LLMCallRecord]:
        out: list[LLMCallRecord] = []
        for llm, m in marks:
            for c in llm.calls[m:]:
                out.append(c.model_copy() if isinstance(c, LLMCallRecord) else LLMCallRecord.model_validate(c))
        return out

    def _llm_label(self, used: list[Any]) -> str:
        names: dict[str, None] = {}
        for llm in used:
            names.setdefault(str(getattr(llm, "name", "llm")), None)
        return ", ".join(names) if names else NO_LLM

    # ------------------------------------------------------------------ steps
    def translate(self, idea: str) -> StrategyTranslation:
        return self.translator.translate(idea)

    def _interpret(self, result: BacktestResult) -> list[EvidenceCheck]:
        """Attach an interpretation to ``result``; the LLM interpreter falls back to the heuristic one."""
        try:
            interp, checks = self.interpreter.interpret(result)
        except Exception as exc:  # noqa: BLE001 - a failed review must not lose the backtest
            if isinstance(self.interpreter, HeuristicInterpreter):
                raise
            result.warnings.append(
                f"the LLM interpretation failed ({type(exc).__name__}: {str(exc)[:200]}); the offline heuristic "
                "interpretation is shown instead"
            )
            interp, checks = HeuristicInterpreter().interpret(result)
        result.interpretation = interp
        return list(checks)

    def _finish(self, result: BacktestResult, marks: list[tuple[Any, int]]) -> None:
        calls = self._new_calls(marks)
        used = [llm for llm, m in marks if len(llm.calls) > m]
        result.llm_calls = [*result.llm_calls, *calls]
        result.llm = self._llm_label(used)

    @staticmethod
    def _template_args(spec: StrategySpec) -> dict[str, Any]:
        t = TEMPLATES.get(spec.name)
        if t is None:
            return {}
        return {"template_title": t.title, "template_description": t.description, "references": list(t.references)}

    def _write(self, result: BacktestResult, spec: StrategySpec, checks: list[EvidenceCheck],
               extra_html: str | None = None) -> tuple[Path | None, Path | None]:
        folder = self.run_dir(result.run_id)
        if folder is None:
            return None, None
        from aitrading.report.html import render_backtest_html

        json_path = folder / "result.json"
        _write_atomic(json_path, result.model_dump_json(indent=2))
        page = render_backtest_html(result, metric_checks=checks or None, **self._template_args(spec))
        if extra_html:
            marker = '<footer class="disclaimer">'
            page = page.replace(marker, extra_html + "\n" + marker, 1) if marker in page else page + extra_html
        html_path = folder / "report.html"
        _write_atomic(html_path, page)
        return html_path, json_path

    # ------------------------------------------------------------------ public API
    def run(self, idea: str | None = None, *, spec: StrategySpec | None = None, interpret: bool = True) -> IdeaLabResult:
        """Translate ``idea`` (unless ``spec`` is given), backtest, interpret and save the run."""
        marks = self._marks()
        translation: StrategyTranslation | None = None
        if spec is None:
            if not idea or not str(idea).strip():
                raise ValueError("give an idea (e.g. 'Fama-French 3 factor model') or a StrategySpec")
            translation = self.translator.translate(str(idea))
            spec = translation.spec
        result = self.runner.backtest(spec)
        checks = self._interpret(result) if interpret else []
        self._finish(result, marks)
        html_path, json_path = self._write(result, spec, checks)
        return IdeaLabResult(spec=spec, translation=translation, result=result, checks=checks,
                             html_path=html_path, json_path=json_path)

    def replicate_idea(
        self,
        candidate: Any,
        spec: StrategySpec,
        *,
        progress: Callable[[str], None] | None = None,
        interpret: bool = True,
        **kwargs: Any,
    ) -> tuple[BacktestResult, Any, Path | None]:
        """Backtest ``spec`` plus the replication / robustness suite; returns ``(base result, report, html path)``.

        ``candidate`` is the inbox idea (``IdeaCandidate``) the spec came from, or ``None``; extra
        keyword arguments go to :func:`~aitrading.discovery.replicate.replicate`.
        """
        from aitrading.discovery.replicate import replicate
        from aitrading.report.html import _replication_html, _section

        marks = self._marks()
        result, report = replicate(candidate, spec, self.runner, progress=progress, **kwargs)
        checks = self._interpret(result) if interpret else []
        self._finish(result, marks)
        section = _section("replication", "Replication and robustness", _replication_html(report),
                           intro="The same strategy re-run on sub-periods, after publication, at other cost levels "
                                 "and with perturbed parameters.")
        html_path, json_path = self._write(result, spec, checks, extra_html=section)
        if json_path is not None:
            _write_atomic(json_path.parent / "replication.json", json.dumps(report.model_dump(mode="json"), indent=2))
        return result, report, html_path
