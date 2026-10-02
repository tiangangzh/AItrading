"""Numbers in an interpretation's prose are compared with the backtest result before a report shows them."""

from __future__ import annotations

import pytest
from test_html_report import make_pipeline
from test_interpret import make_result

from aitrading.backtest.models import BacktestInterpretation, CitedMetric
from aitrading.report.html import render_backtest_html, render_pipeline_html
from aitrading.report.prose import interpretation_unverified_numbers, stat_numbers, unverified_numbers
from aitrading.strategy.interpret import HeuristicInterpreter


@pytest.mark.parametrize("text, expected", [
    ("Sharpe of 1.4 with a t-stat of 3.2 - robust.", ["1.4", "3.2"]),
    ("FCF yield of 9% vs sector 4%; trades at 12.3x with 25 bps costs and a 0.6pp margin gain", ["9%", "4%", "12.3x", "25 bps", "0.6pp"]),
    ("Revenue of $1.08 billion and a $20B cap", ["1.08 billion", "20B"]),
    # years, counts, look-backs, identifiers and URLs are not claims
    ("12-1 momentum, 200-day average, Q3 2025, FF5, 10-Q, 8-K, x12, 12m return, 3rd decile, 2026-09-30, "
     "arxiv.org/abs/2601.00001, v1.2, 5 buckets", []),
    ("pulled back 15-40% and ends at 29.55.", ["40%", "29.55"]),
])
def test_stat_numbers(text, expected):
    assert [n.text for n in stat_numbers(text)] == expected


def test_unverified_numbers_allow_rounding_sign_and_scale():
    known = [0.4213, -28.1, 0.031, 2.47]
    assert unverified_numbers(["Sharpe 0.42, fell 28%, IC of 3.1%, cap $2.5bn"], known) == []
    assert unverified_numbers(["Sharpe 0.5 and 1.4, IC 0.04"], known) == ["0.5", "1.4", "0.04"]
    assert unverified_numbers(["2,470 million"], [2.47]) == []


def _fabricated(cited: list[CitedMetric] | None = None) -> BacktestInterpretation:
    return BacktestInterpretation(summary="Sharpe of 1.4 with a t-stat of 3.2 - robust.", verdict="promising",
                                  key_findings=["Alpha t-stat 3.2, Sharpe 1.4"], cited_metrics=cited or [],
                                  biases_and_caveats=[], next_experiments=[])


def _verdict(page: str) -> str:
    return page[page.index('id="verdict"'):page.index("</section>", page.index('id="verdict"'))]


def test_fabricated_numbers_in_an_uncited_interpretation_are_flagged_in_the_report():
    res = make_result(sharpe=0.42, alpha_t=1.6, mono=0.5)
    res = res.model_copy(update={"interpretation": _fabricated()})
    assert interpretation_unverified_numbers(res) == ["1.4", "3.2"]
    section = _verdict(render_backtest_html(res))
    assert "Unverified numbers in the text above" in section
    assert section.count('aria-label="unverified"') == 2 and "1.4" in section and "3.2" in section
    assert "cited no numbers for verification" in section


def test_heuristic_interpretation_has_no_unverified_numbers():
    # n_periods 130 / 30 (monthly): years of observations differ from the 2015-01-31..2024-12-31 calendar
    # span, as daily periods / 252 do on the synthetic market's holiday-free calendar (regression: the
    # offline verdict's "Over 10.3 years (2016-09-30 to 2026-09-30)" was flagged UNVERIFIED by the CLI)
    for kw in ({}, {"sharpe": 0.42, "alpha_t": 1.6, "mono": 0.5}, {"regression": False}, {"quantiles": False},
               {"n_periods": 130}, {"n_periods": 30}):
        res = make_result(**kw)
        res = res.model_copy(update={"interpretation": HeuristicInterpreter().interpret(res)[0]})
        assert interpretation_unverified_numbers(res) == [], kw
        assert "Unverified numbers" not in _verdict(render_backtest_html(res))


def test_pipeline_report_says_narrative_numbers_are_not_checked():
    page = render_pipeline_html(make_pipeline())
    card = page[page.index('id="idea-ACME"'):page.index('id="idea-BETA"')]
    assert "numbers in the thesis text above, the catalysts and the risks are not individually verified" in card
    assert "Numbers in the narrative text are not individually checked." in page
