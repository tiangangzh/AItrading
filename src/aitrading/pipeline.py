"""End-to-end research pipeline: observation -> screen -> ranked candidates -> grounded explanations.

``ResearchPipeline.run`` performs, in order:

1. A run id: ``YYYYMMDD-<12 hex of sha256(observation, as_of, provider name)>``, with ``-2``,
   ``-3``, ... appended when ``<out_dir>/<run_id>`` already exists (the directory is claimed at the
   start of the run and removed again if the run fails before writing anything).
2. Translation of the observation into a ``ScreenSpec`` (skipped when ``spec`` is given), an optional
   ``top_n`` override, and validation against the feature catalog before any data is fetched.
3. Vendor push-down: if the provider implements ``ScreenPushdown`` the spec is pushed down to
   narrow the universe (the query is recorded; a failure is a warning and the full universe is
   screened). The local engine re-evaluates every condition on the narrowed set.
4. ``provider.get_universe`` and a screen-pass ``FeatureEngine.build`` for the spec's features, the
   universe-filter features, ``price``, ``gics_sector``, ``market_cap_usd_bn`` and the
   cross-sectional features; then ``run_screen`` and ``rank_candidates``.
5. Enrichment: a full-catalog ``FeatureEngine.build`` for the ranked candidates only (cross-sectional
   features are taken from the screen pass, where they are relative to the whole universe).
6. For the first ``explain_top_k`` candidates: point-in-time documents (``gather_documents``, which
   applies the provider's ``DataBoundary``), budgeted verbatim excerpts, the rendered documents
   block, a narrative-signal tally and sector medians (screen-pass rows of the candidate's
   ``gics_sector`` that pass the universe filters; omitted when push-down narrowed the universe),
   then ``explainer.explain``. ``LLMError`` / ``LLMRefusalError`` (and ``ProviderError``, or a
   pydantic ``ValidationError`` for unusable structured output) are recorded in
   ``InvestmentIdea.error`` and the run continues. Candidates ranked below
   ``explain_top_k`` are listed without a thesis.
7. ``PipelineResult`` with the funnel, coverage of the spec's features, LLM call records made during
   this run (from ``translator.llm`` / ``explainer.llm`` when present) and warnings (feature-engine
   degradations, push-down failures, unsupported requests, withheld documents, repair errors, and
   messages the provider appended to its own ``warnings`` list during the run).
8. If ``out_dir`` is set: ``result.json``, ``spec.json``, ``features.csv`` (screen-pass rows of the
   survivors) and ``documents.json`` (doc_id, kind, title, published_at, source per explained idea;
   never document text) under ``<out_dir>/<run_id>/``.

Data boundary: only documents the provider's boundary permits are excerpted (re-checked here after
``gather_documents``), each excerpt is capped at ``boundary.max_chars_per_document``, and when
``boundary.allow_numeric_features`` is False an LLM-backed explainer (one with an ``llm``
attribute) receives no feature values.

Determinism: everything except ``started_at`` / ``finished_at`` (wall clock, injectable via
``clock``) is a function of the inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

import pandas as pd
from pydantic import ValidationError

from aitrading.core.models import Document, FunnelStep, InvestmentIdea, LLMCallRecord, PipelineResult, RankedCandidate
from aitrading.core.policy import DataBoundary
from aitrading.data.base import MarketDataProvider, ProviderError, ScreenPushdown
from aitrading.llm.base import LLMError
from aitrading.narrative.excerpts import build_bundle_excerpts, render_documents_for_prompt
from aitrading.narrative.retrieval import NarrativeBundle, gather_documents
from aitrading.narrative.signals import extract_signals_from_documents, summarize_signals
from aitrading.rank.scoring import rank_candidates
from aitrading.screen.catalog import FeatureCatalog, default_catalog
from aitrading.screen.engine import LIQUIDITY_COLUMN, PRICE_COLUMN, SECTOR_COLUMN, ScreenValidationError, apply_universe, run_screen
from aitrading.screen.features import CROSS_SECTIONAL_FEATURES, FeatureEngine
from aitrading.screen.spec import ScreenSpec, UniverseSpec

__all__ = ["ResearchPipeline", "make_run_id", "NO_LLM"]

NO_LLM = "none (offline heuristics)"


class _Translator(Protocol):
    def translate(self, observation: str) -> Any: ...  # TranslationResult (``.spec``) or a ScreenSpec


class _Explainer(Protocol):
    def explain(self, candidate, features, documents, documents_prompt, spec, as_of, sector_context=None, signal_summary=None) -> Any: ...


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_date(d: date | datetime | str) -> date:
    if isinstance(d, str):
        return date.fromisoformat(d[:10])
    return d.date() if isinstance(d, datetime) else d


def make_run_id(observation: str, as_of: date, provider_name: str) -> str:
    """``YYYYMMDD-<12 hex>``: a stable id for (observation, as_of, provider)."""
    as_of = _as_date(as_of)
    payload = "\x1f".join([observation, as_of.isoformat(), provider_name]).encode("utf-8")
    return f"{as_of:%Y%m%d}-{hashlib.sha256(payload).hexdigest()[:12]}"


def _py(value: Any) -> float | str | None:
    """Feature value for a candidate's feature dict: finite float, label text, or None."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    return x if math.isfinite(x) else None


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


class ResearchPipeline:
    """Observation in, ``PipelineResult`` out (see the module docstring for the steps)."""

    def __init__(
        self,
        provider: MarketDataProvider,
        translator: _Translator | None,
        explainer: _Explainer | None,
        *,
        catalog: FeatureCatalog | None = None,
        out_dir: str | Path | None = "runs",
        explain_top_k: int = 5,
        documents_lookback_days: int = 120,
        max_doc_chars_total: int = 40_000,
        clock: Callable[[], datetime] | None = None,
    ):
        if explain_top_k < 0:
            raise ValueError("explain_top_k must be >= 0")
        if documents_lookback_days < 0:
            raise ValueError("documents_lookback_days must be >= 0")
        if max_doc_chars_total < 0:
            raise ValueError("max_doc_chars_total must be >= 0")
        self.provider = provider
        self.translator = translator
        self.explainer = explainer
        self.catalog = catalog or default_catalog()
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.explain_top_k = explain_top_k
        self._capped_documents = 0
        self.documents_lookback_days = documents_lookback_days
        self.max_doc_chars_total = max_doc_chars_total
        self.clock = clock or _utcnow
        self.engine = FeatureEngine(provider, self.catalog)

    # -- helpers -----------------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return str(getattr(self.provider, "name", type(self.provider).__name__))

    @property
    def boundary(self) -> DataBoundary:
        """The provider's boundary (the same object ``gather_documents`` applies), else the default."""
        b = getattr(self.provider, "boundary", None)
        return b if b is not None and hasattr(b, "permits") else DataBoundary(provider=self.provider_name)

    def _llms(self) -> list[Any]:
        """Distinct LLM objects (attribute ``llm`` with ``calls``) of the translator and explainer."""
        out: list[Any] = []
        for owner in (self.translator, self.explainer):
            llm = getattr(owner, "llm", None)
            if llm is not None and isinstance(getattr(llm, "calls", None), list) and all(llm is not o for o in out):
                out.append(llm)
        return out

    def _capped_note(self) -> list[str]:
        """One summary line for documents skipped only because of the per-ticker document cap
        (routine, not a data-quality problem), instead of one warning per document."""
        n, self._capped_documents = self._capped_documents, 0
        if not n:
            return []
        cap = getattr(self.boundary, "max_documents_per_ticker", "?")
        return [f"{n} older or lower-priority document(s) were not shown to the explainer (limit {cap} per ticker; see documents.json)"]

    def _claim_run_dir(self, base: str) -> tuple[str, Path | None]:
        if self.out_dir is None:
            return base, None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        k = 1
        while True:
            run_id = base if k == 1 else f"{base}-{k}"
            path = self.out_dir / run_id
            try:
                path.mkdir()
                return run_id, path
            except FileExistsError:
                k += 1

    @staticmethod
    def _screen_features(spec: ScreenSpec) -> set[str]:
        u: UniverseSpec = spec.universe
        out = set(spec.features()) | {PRICE_COLUMN, SECTOR_COLUMN, "market_cap_usd_bn"} | set(CROSS_SECTIONAL_FEATURES)
        if u.min_avg_dollar_volume_usd_mn is not None:
            out.add(LIQUIDITY_COLUMN)
        return out

    @staticmethod
    def _spec_feature_order(spec: ScreenSpec) -> list[str]:
        names: list[str] = []
        for c in spec.all_conditions():
            names.append(c.feature)
            if c.other_feature:
                names.append(c.other_feature)
        names += [f.feature for f in spec.ranking]
        return _dedupe(names)

    def _sector_medians(self, frame: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
        numeric = [f for f in self.catalog.names() if f in frame.columns and self.catalog[f].dtype != "category"]
        pool = frame.loc[mask.to_numpy(), [SECTOR_COLUMN, *numeric]]
        pool = pool[pool[SECTOR_COLUMN].notna()]
        if pool.empty:
            return pd.DataFrame(columns=numeric, dtype="float64")
        return pool.groupby(SECTOR_COLUMN, sort=True)[numeric].median()

    # -- run ---------------------------------------------------------------------------------

    def run(self, observation: str, as_of: date, *, top_n: int | None = None, spec: ScreenSpec | None = None) -> PipelineResult:
        """Run the pipeline once; raises on translation / validation / required-data failures."""
        as_of = _as_date(as_of)
        started_at = self.clock()
        llms = self._llms()
        marks = [len(llm.calls) for llm in llms]
        provider_warnings = getattr(self.provider, "warnings", None)
        mark = len(provider_warnings) if isinstance(provider_warnings, list) else None
        run_id, run_dir = self._claim_run_dir(make_run_id(observation, as_of, self.provider_name))
        try:
            result, artifacts = self._run(run_id, observation, as_of, top_n, spec, started_at, llms, marks, mark)
        except BaseException:
            if run_dir is not None:
                try:
                    os.rmdir(run_dir)
                except OSError:
                    pass
            raise
        if run_dir is not None:
            self._write(run_dir, result, *artifacts)
        return result

    def _run(
        self,
        run_id: str,
        observation: str,
        as_of: date,
        top_n: int | None,
        spec: ScreenSpec | None,
        started_at: datetime,
        llms: list[Any],
        marks: list[int],
        provider_mark: int | None,
    ) -> tuple[PipelineResult, tuple[ScreenSpec, pd.DataFrame, list[dict[str, Any]]]]:
        warnings: list[str] = []

        # 1. spec
        if spec is None:
            if self.translator is None:
                raise ValueError("a translator is required when no spec is given")
            out = self.translator.translate(observation)
            spec = out.spec if hasattr(out, "spec") else out
            rounds = getattr(out, "errors_by_round", None) or []
            if len(rounds) > 1:
                warnings.append(f"screen translation needed {len(rounds) - 1} repair round(s): " + "; ".join(e for r in rounds[:-1] for e in r)[:500])
        if not isinstance(spec, ScreenSpec):
            raise TypeError(f"translator returned {type(spec).__name__}, expected ScreenSpec")
        if top_n is not None:
            spec = ScreenSpec.model_validate({**spec.model_dump(), "top_n": top_n})
        errors = spec.validate_against(self.catalog)
        if errors:
            raise ScreenValidationError(errors)
        warnings += [f"not screened (no catalog feature expresses it): {r}" for r in spec.unsupported_requests]

        # 2. push-down
        pushdown_query: str | None = None
        pushed: list[str] | None = None
        if isinstance(self.provider, ScreenPushdown):
            try:
                pr = self.provider.pushdown_screen(spec, as_of)
                query, tickers = str(pr.query), _dedupe([str(t) for t in pr.tickers])
                pushdown_query, pushed = query, tickers
            except Exception as exc:  # noqa: BLE001 - push-down is an optimisation; fall back to local
                warnings.append(f"screen push-down failed ({type(exc).__name__}: {exc}); screening the full universe locally")

        # 3. universe + screen pass
        universe = self.provider.get_universe(spec.universe, as_of)
        universe = universe[~universe.index.duplicated(keep="first")]
        universe_size = len(universe)
        head: list[FunnelStep] = []
        if pushed is not None:
            inside = set(map(str, universe.index))
            outside = [t for t in pushed if t not in inside]
            if outside:
                warnings.append(f"push-down returned {len(outside)} ticker(s) outside the universe; ignored")
            universe = universe[universe.index.map(str).isin(set(pushed))]
            head.append(FunnelStep(label=f"vendor push-down ({self.provider_name})", passed_alone=len(universe), remaining=len(universe)))

        screen = self.engine.build(universe, as_of, self._screen_features(spec))
        warnings += screen.warnings
        frame = screen.frame
        outcome = run_screen(spec, frame, self.catalog)
        funnel = head + outcome.funnel
        if not outcome.survivors:
            warnings.append("no names passed the screen")
        spec_order = self._spec_feature_order(spec)
        coverage = {f: screen.coverage.get(f, 0.0) for f in spec_order}
        candidates = rank_candidates(frame, spec.ranking, outcome.survivors, spec.top_n, extra_features=spec_order)

        # 4. enrichment for the ranked candidates
        top = [c.ticker for c in candidates]
        rich_frame = frame.loc[top] if top else frame.iloc[:0]
        if top:
            try:
                rich = self.engine.build(screen.universe.loc[top], as_of, None)
                rich_frame = rich.frame
                for f in CROSS_SECTIONAL_FEATURES:
                    rich_frame[f] = frame.loc[top, f]
                warnings += rich.warnings
            except ProviderError as exc:
                warnings.append(f"enrichment failed ({exc}); candidates explained with screen-pass features only")

        # 5. explanations
        medians: pd.DataFrame | None = None
        explaining = bool(top) and self.explain_top_k > 0 and self.explainer is not None
        if explaining and pushed is None:
            umask, _ = apply_universe(spec.universe, frame)
            medians = self._sector_medians(frame, umask)
        elif explaining:
            warnings.append("sector medians omitted: push-down narrowed the universe, so they would not be representative")
        ideas: list[InvestmentIdea] = []
        documents_log: list[dict[str, Any]] = []
        for i, cand in enumerate(candidates):
            if i >= self.explain_top_k or self.explainer is None:
                ideas.append(InvestmentIdea(candidate=cand))
                continue
            idea, docs = self._explain(cand, rich_frame, medians, spec, as_of, warnings)
            ideas.append(idea)
            documents_log.append(
                {
                    "rank": cand.rank,
                    "ticker": cand.ticker,
                    "documents": [
                        {
                            "doc_id": d.doc_id,
                            "kind": d.kind.value,
                            "title": d.title,
                            "published_at": d.published_at.isoformat(),
                            "source": d.source,
                        }
                        for d in docs
                    ],
                }
            )

        if provider_mark is not None:  # adapters (free, Bloomberg, LSEG) collect their caveats in .warnings
            warnings += [f"provider {self.provider_name}: {w}" for w in self.provider.warnings[provider_mark:]]
        calls: list[LLMCallRecord] = [c.model_copy() for llm, m in zip(llms, marks) for c in llm.calls[m:]]
        names = _dedupe([str(getattr(llm, "name", "llm")) for llm in llms])
        result = PipelineResult(
            run_id=run_id,
            observation=observation,
            as_of=as_of,
            provider=self.provider_name,
            llm=", ".join(names) if names else NO_LLM,
            spec=spec.model_dump(mode="json"),
            universe_size=universe_size,
            feature_coverage=coverage,
            funnel=funnel,
            survivors=len(outcome.survivors),
            ideas=ideas,
            llm_calls=calls,
            pushdown_query=pushdown_query,
            warnings=_dedupe(warnings + self._capped_note()),
            started_at=started_at,
            finished_at=self.clock(),
        )
        survivors_frame = frame.loc[outcome.survivors]
        return result, (spec, survivors_frame, documents_log)

    def _explain(
        self,
        cand: RankedCandidate,
        rich_frame: pd.DataFrame,
        medians: pd.DataFrame | None,
        spec: ScreenSpec,
        as_of: date,
        warnings: list[str],
    ) -> tuple[InvestmentIdea, list[Document]]:
        ticker = cand.ticker
        boundary = self.boundary
        row = rich_frame.loc[ticker]
        features: dict[str, float | str | None] = {f: _py(row[f]) for f in self.catalog.names() if f in rich_frame.columns}
        sector = features.get(SECTOR_COLUMN)
        context: dict[str, float | None] | None = None
        if medians is not None and isinstance(sector, str) and sector in medians.index:
            med = medians.loc[sector]
            context = {f: float(med[f]) for f in features if f in med.index and pd.notna(med[f])}

        bundle = gather_documents(self.provider, ticker, as_of, lookback_days=self.documents_lookback_days)
        permitted = [d for d in bundle.documents if boundary.permits(d)]
        for d in bundle.documents:
            if not boundary.permits(d):  # gather_documents already filters; never let one through
                warnings.append(f"{ticker}: {d.doc_id} dropped: {d.kind.value} not permitted by the data boundary")
        capped = [w for w in bundle.withheld if "over max_documents_per_ticker" in w]
        warnings += [f"{ticker}: withheld {w}" for w in bundle.withheld if w not in capped]
        self._capped_documents += len(capped)
        excerpts = build_bundle_excerpts(
            NarrativeBundle(ticker=ticker, documents=permitted, withheld=list(bundle.withheld)),
            max_chars_total=self.max_doc_chars_total,
            per_doc_max=getattr(boundary, "max_chars_per_document", None),
        )
        shown_ids = [ex.doc_id for ex in excerpts]
        docs = [d for d in permitted if d.doc_id in set(shown_ids)]
        documents_prompt = render_documents_for_prompt(excerpts)
        summary = summarize_signals(extract_signals_from_documents(docs))

        if getattr(self.explainer, "llm", None) is not None and not getattr(boundary, "allow_numeric_features", True):
            warnings.append(f"feature values withheld from the explainer: the data boundary of '{self.provider_name}' forbids numeric features")
            features, context = {}, None

        try:
            res = self.explainer.explain(cand, features, docs, documents_prompt, spec, as_of, sector_context=context, signal_summary=summary)
        except (LLMError, ProviderError, ValidationError) as exc:  # ValidationError: unusable structured output
            return InvestmentIdea(candidate=cand, documents_used=shown_ids, error=f"{type(exc).__name__}: {exc}"), docs
        if getattr(res, "repair_error", None):
            warnings.append(f"{ticker}: repair round failed ({res.repair_error}); kept the first thesis")
        idea = InvestmentIdea(candidate=cand, thesis=res.thesis, grounding=res.grounding, documents_used=shown_ids)
        return idea, docs

    # -- artifacts ---------------------------------------------------------------------------

    @staticmethod
    def _write(run_dir: Path, result: PipelineResult, spec: ScreenSpec, survivors: pd.DataFrame, documents: list[dict[str, Any]]) -> None:
        (run_dir / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        (run_dir / "spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        survivors.to_csv(run_dir / "features.csv", index=True, index_label="ticker")
        (run_dir / "documents.json").write_text(json.dumps(documents, indent=2), encoding="utf-8")
