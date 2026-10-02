# Equity research report: mid_cap_uptrend_momentum_pullback

**Investment observation**

> Find US mid-caps ($2-20B) that were in established uptrends (50-day above 200-day, positive 12-1 momentum) but have pulled back 15-40% from their 52-week highs over the past few months on heavy volume, now oversold (RSI under 40), still generating strong free cash flow (FCF yield above 4%) with revenue growth above 8%, and where short interest is elevated (above 6% of float). Rank by FCF yield, growth and the size of the drawdown, then read the latest earnings calls and explain the dislocation.

| Run | Value |
|---|---|
| As of | 2026-09-30 |
| Provider | synthetic |
| LLM | none (offline heuristics) |
| Run id | `20260930-8b70f6bf1e2b` |
| Started | 2026-10-02 09:00:42 UTC |
| Finished | 2026-10-02 09:00:43 UTC (1.2 s) |
| Universe | 449 names |
| Passed the screen | 14 |
| Ranked / explained | 10 / 5 |

## Summary

449 names were in the universe as of 2026-09-30; 14 passed all 9 condition(s); the top 10 were ranked by `fcf_yield_pct`, `revenue_growth_yoy_pct`, `drawdown_from_52w_high_pct`. 5 candidate(s) were explained: 4 actionable, 5 with every evidence item verified against the source data. 11 warning(s) are listed in appendix A.3.

| Rank | Ticker | Name | Dislocation type | Conviction | Actionable | Grounding |
|---|---|---|---|---|---|---|
| #1 | DLCR | Delacroix Storage | Transitory fundamental shock | high | yes | ✓ 19/19 |
| #2 | MARM | Marrowgate Medical | Transitory fundamental shock | high | yes | ✓ 19/19 |
| #3 | GOLF | Goldcrest Facility Services | Transitory fundamental shock | high | yes | ✓ 19/19 |
| #4 | STON | Stonebridge Diagnostics | Transitory fundamental shock | high | yes | ✓ 19/19 |
| #5 | THOI | Thornbury Industries | Structural decline value trap | high | no | ✓ 19/19 |

## 1. Screen specification

Screen `mid_cap_uptrend_momentum_pullback`; top 10 survivors are ranked.

### Universe

| Filter | Setting |
|---|---|
| Country | US |
| Security types | common_stock |
| Minimum price | $5 |
| Minimum 20-day avg dollar volume | $5mn |
| Excluded sectors | none |

### Conditions (all must hold)

| # | Condition | Rationale |
|---:|---|---|
| 1 | `market_cap_usd_bn between 2 and 20` | '$2-20B' |
| 2 | `sma_50_vs_sma_200_pct > 0` | '50-day above 200-day' |
| 3 | `return_12m_ex_1m_pct > 0` | 'positive 12-1 momentum' |
| 4 | `drawdown_from_52w_high_pct between -40 and -15` | 'pulled back 15-40% from their 52-week highs' |
| 5 | `max_volume_ratio_20d >= 2` | 'heavy volume' (default for 'heavy volume') |
| 6 | `rsi_14 < 40` | 'RSI under 40' |
| 7 | `fcf_yield_pct > 4` | 'FCF yield above 4%' |
| 8 | `revenue_growth_yoy_pct > 8` | 'revenue growth above 8%' |
| 9 | `short_interest_pct_float > 6` | 'short interest is elevated (above 6% of float)' |

### Ranking

| Factor | Direction | Weight | Rationale |
|---|---|---:|---|
| `fcf_yield_pct` | higher is better | 33% | rank by 'fcf yield' |
| `revenue_growth_yoy_pct` | higher is better | 33% | rank by 'growth' |
| `drawdown_from_52w_high_pct` | lower is better | 33% | rank by 'the size of the drawdown' |

### Assumptions

- Left to the narrative / explanation stages (not a screen condition): 'then read the latest earnings calls and explain the dislocation'.
- 'heavy volume' read as max_volume_ratio_20d >= 2 (default).
- Universe: platform defaults (US common stock, price >= $5, 20-day average dollar volume >= $5mn).

### Unsupported requests

_None: every part of the observation was expressed as a condition, ranking factor or assumption._

## 2. Screen funnel

Universe filters and conditions are applied cumulatively, in order. _Passed alone_ counts names satisfying the step on its own; _missing data_ counts names excluded because the feature was missing.

| # | Step | Passed alone | Remaining | Missing data |
|---:|---|---:|---:|---:|
| 0 | Universe | 449 | 449 |  |
| 1 | country == US | 449 | 449 | 0 |
| 2 | security_type in [common_stock] | 449 | 449 | 0 |
| 3 | price >= 5 | 441 | 441 | 0 |
| 4 | avg_dollar_volume_20d_usd_mn >= 5 | 421 | 418 | 0 |
| 5 | market_cap_usd_bn between 2 and 20 | 258 | 258 | 0 |
| 6 | sma_50_vs_sma_200_pct > 0 | 296 | 163 | 0 |
| 7 | return_12m_ex_1m_pct > 0 | 225 | 115 | 0 |
| 8 | drawdown_from_52w_high_pct between -40 and -15 | 192 | 42 | 0 |
| 9 | max_volume_ratio_20d >= 2 | 144 | 33 | 0 |
| 10 | rsi_14 < 40 | 56 | 24 | 0 |
| 11 | fcf_yield_pct > 4 | 244 | 20 | 0 |
| 12 | revenue_growth_yoy_pct > 8 | 220 | 19 | 0 |
| 13 | short_interest_pct_float > 6 | 110 | 14 | 0 |

**14 name(s) passed every step.**

## 3. Feature coverage

Share of the universe with a value for each feature the screen uses. A missing value never satisfies a condition, so low coverage silently shrinks the candidate set.

| Feature | Unit | Coverage | Status |
|---|---|---:|---|
| `market_cap_usd_bn` | USD bn | 100.0% | ok |
| `sma_50_vs_sma_200_pct` | % | 99.6% | ok |
| `return_12m_ex_1m_pct` | % | 99.6% | ok |
| `drawdown_from_52w_high_pct` | % | 99.6% | ok |
| `max_volume_ratio_20d` | x | 100.0% | ok |
| `rsi_14` | 0-100 | 100.0% | ok |
| `fcf_yield_pct` | % | 100.0% | ok |
| `revenue_growth_yoy_pct` | % | 100.0% | ok |
| `short_interest_pct_float` | % | 100.0% | ok |

All screen features have at least 90% coverage.

## 4. Ranked candidates

Score = weighted mean of each ranking factor's percentile among the survivors (0-1, higher is better).

| Rank | Ticker | Name | Score | fcf_yield_pct (%) | revenue_growth_yoy_pct (%) | drawdown_from_52w_high_pct (%) | market_cap_usd_bn (USD bn) | sma_50_vs_sma_200_pct (%) | return_12m_ex_1m_pct (%) | max_volume_ratio_20d (x) | rsi_14 (0-100) | short_interest_pct_float (%) | Explained |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | DLCR | Delacroix Storage | 0.821 | 8.206 | 14.29 | -28.1 | 10.22 | 1.474 | 48.87 | 4.28 | 29.55 | 8.538 | yes |
| 2 | MARM | Marrowgate Medical | 0.692 | 7.908 | 13.7 | -28.85 | 4.28 | 6.924 | 19.24 | 3.116 | 28.48 | 19.63 | yes |
| 3 | GOLF | Goldcrest Facility Services | 0.667 | 8.017 | 18.43 | -21.01 | 6.68 | 12.73 | 47.56 | 2.815 | 35.28 | 11.58 | yes |
| 4 | STON | Stonebridge Diagnostics | 0.590 | 8.201 | 15.02 | -20.02 | 3.371 | 11.71 | 30.47 | 3.752 | 31.82 | 13.92 | yes |
| 5 | THOI | Thornbury Industries | 0.590 | 7.612 | 10.55 | -31.06 | 6.688 | 3.849 | 23.34 | 3.658 | 29.2 | 10.87 | yes |
| 6 | INGD | Ingleby Defense Systems | 0.462 | 6.254 | 13.71 | -23.58 | 3.615 | 12.64 | 42.45 | 2.936 | 34.49 | 7.491 | no |
| 7 | IVOC | Ivorytown Cloud | 0.462 | 6.902 | 10.22 | -29.2 | 13.23 | 4.967 | 43.75 | 6.204 | 32.07 | 12.74 | no |
| 8 | BLAC | Blackthorn Digital | 0.436 | 7.899 | 10.89 | -22.01 | 4.057 | 14.14 | 52.87 | 10.32 | 30.41 | 10.8 | no |
| 9 | JESS | Jessamine Medical | 0.436 | 6.663 | 13.7 | -22.12 | 6.69 | 18.08 | 48.87 | 3.668 | 32.7 | 8.785 | no |
| 10 | VELB | Vellum Broadcasting | 0.436 | 6.116 | 10.48 | -31.14 | 4.512 | 1.996 | 52.26 | 3.801 | 33.71 | 18.02 | no |

## 5. Investment ideas

Every quoted sentence and every cited number below was checked programmatically against the source documents and the feature table (✓ verified, ✗ mismatch or not found, – not checked).

### 5.1 #1 DLCR - Delacroix Storage

**Thesis:** Delacroix Storage (DLCR) looks like a transitory shock: the shares show a -28.1% drawdown from the 52-week high and an RSI of 29.55, yet management frames the hit as temporary and latest-quarter revenue growth of 8.906% (14.29% trailing) is still positive.

- **Dislocation type:** Transitory fundamental shock (`transitory_fundamental_shock`)
- **Actionable:** yes · **Conviction:** high
- **Rank score:** 0.821 (`fcf_yield_pct` 1.00, `revenue_growth_yoy_pct` 0.77, `drawdown_from_52w_high_pct` 0.69)
- **Grounding:** 19/19 evidence items verified (100%) - fully grounded
- **Documents shown to the explainer:** `SYN-TR-DLCR-20260803`, `SYN-NW-DLCR-20260918-RATING`, `SYN-NW-DLCR-20260803-MOVE`, `SYN-NW-DLCR-20260803-PR`

#### Market narrative vs variant view

**Market narrative.** With a -28.1% drawdown from the 52-week high and an RSI of 29.55, and short interest at 8.538% of float (19.71% change over one month), the price implies the market expects the problem behind the last report to persist.

**Variant view.** The evidence points to a one-off rather than a broken business: narrative signals transitory_language x12, no_share_loss x1, no_customer_churn x1, no_pricing_pressure x1, demand_resilience x7, backlog_strength x3; a -7.679% revision to NTM EPS consensus over 3 months is a trim, not a collapse.

#### Why the dislocation exists

Holders of a former uptrend sold the miss on heavy volume and trailing-growth screens extrapolated it. If the next report shows the one-off reversing, the gap can close; if the issue recurs, the market was right.

#### Quantitative evidence

| Check | Feature | Value | Interpretation |
|---|---|---:|---|
| ✓ verified | `market_cap_usd_bn` | 10.22 | screen condition market_cap_usd_bn between 2 and 20: passes; sector median 7.29 |
| ✓ verified | `sma_50_vs_sma_200_pct` | 1.474 | screen condition sma_50_vs_sma_200_pct > 0: passes; sector median 5.766 |
| ✓ verified | `return_12m_ex_1m_pct` | 48.87 | screen condition return_12m_ex_1m_pct > 0: passes; sector median 17.65 |
| ✓ verified | `drawdown_from_52w_high_pct` | -28.1 | screen condition drawdown_from_52w_high_pct between -40 and -15: passes; sector median -21.68 |
| ✓ verified | `max_volume_ratio_20d` | 4.28 | screen condition max_volume_ratio_20d >= 2: passes; sector median 1.644 |
| ✓ verified | `rsi_14` | 29.55 | screen condition rsi_14 < 40: passes; sector median 56.71 |
| ✓ verified | `fcf_yield_pct` | 8.206 | screen condition fcf_yield_pct > 4: passes; sector median 3.47 |
| ✓ verified | `revenue_growth_yoy_pct` | 14.29 | screen condition revenue_growth_yoy_pct > 8: passes; sector median 11.21 |
| ✓ verified | `short_interest_pct_float` | 8.538 | screen condition short_interest_pct_float > 6: passes; sector median 3.669 |
| ✓ verified | `revenue_growth_last_q_yoy_pct` | 8.906 | latest quarter revenue still growing year on year; sector median 9.45 |
| ✓ verified | `eps_revision_3m_pct` | -7.679 | modest trim to next-12m EPS consensus over 3 months |
| ✓ verified | `revenue_revision_3m_pct` | -7.052 | revenue consensus trimmed |
| ✓ verified | `operating_margin_change_yoy_pp` | -0.6492 | operating margin broadly stable year on year; sector median -0.1751 |
| ✓ verified | `last_eps_surprise_pct` | -3.398 | last quarter missed EPS consensus |

#### Narrative evidence

> ✓ “Revenue of $1.08 billion grew 8.9% year over year, but that figure absorbs a headwind of approximately $66.6 million from the one-time ERP cut-over.”
>
> — Desmond Prescott · `SYN-TR-DLCR-20260803` · grounding: ✓ verified
>
> _the hit is framed as one-off or temporary (transcript, Desmond Prescott; tag transitory_language)_

> ✓ “Order intake was unaffected and actually grew 8% in the quarter; the issue was entirely on the shipping side.”
>
> — Desmond Prescott · `SYN-TR-DLCR-20260803` · grounding: ✓ verified
>
> _underlying demand is described as intact (transcript, Desmond Prescott; tag demand_resilience)_

> ✓ “Our backlog ended the quarter at $1.97 billion, up 10.0% year over year, and book-to-bill was 1.14.”
>
> — Desmond Prescott · `SYN-TR-DLCR-20260803` · grounding: ✓ verified
>
> _backlog or orders are described as strong (transcript, Desmond Prescott; tag backlog_strength)_

> ✓ “Price realization was positive 2.6% in the quarter, and we did not lose a single top-15 customer.”
>
> — Desmond Prescott · `SYN-TR-DLCR-20260803` · grounding: ✓ verified
>
> _management says customers are not leaving (transcript, Desmond Prescott; tag no_customer_churn)_

> ✓ “Gross margin was 58.5%, down 114 basis points from a year ago, reflecting lower absorption on the reduced volume, which we expect to reverse as volumes normalize.”
>
> — Patrick Fairbanks · `SYN-TR-DLCR-20260803` · grounding: ✓ verified
>
> _Cuts against the thesis: margins are under pressure (transcript, Patrick Fairbanks; tag margin_pressure)_

#### Catalysts

- Next earnings report in about 29 days (days_to_next_earnings).
- Next quarter's results showing the one-off item reversing, as management said it would.
- Share repurchases or dividends that management cited.

#### Risks

- Bearish signals in the documents: margin_pressure x2, results_miss x3.
- Short interest at 8.538% of float can extend the decline if the next print disappoints.
- The issue management calls temporary or limited proves larger or recurring.

#### Invalidation triggers

- revenue_growth_last_q_yoy_pct turns negative in the next report.
- eps_revision_3m_pct falls to -15 or below.
- Management lowers or withdraws guidance, or stops describing the issue as temporary.
- Price makes new lows below the current drawdown of -28.1% while short interest keeps rising.

#### Data gaps

- Heuristic baseline: dislocation type, conviction and text come from transparent rules over feature thresholds and phrase-matched narrative signals (aitrading.agent.offline), not from an analyst's or an LLM's judgement; use it as a starting point.
- Decision trace: transitory_fundamental_shock scored 9 (narrative 7.5, quant 1.5); runner-up structural_decline_value_trap scored 2.5

**Grounding score:** 19/19 evidence items verified (100%) - fully grounded.

### 5.2 #2 MARM - Marrowgate Medical

**Thesis:** Marrowgate Medical (MARM) looks like a transitory shock: the shares show a -28.85% drawdown from the 52-week high and an RSI of 28.48, yet management frames the hit as temporary and latest-quarter revenue growth of 7.389% (13.7% trailing) is still positive.

- **Dislocation type:** Transitory fundamental shock (`transitory_fundamental_shock`)
- **Actionable:** yes · **Conviction:** high
- **Rank score:** 0.692 (`fcf_yield_pct` 0.77, `revenue_growth_yoy_pct` 0.54, `drawdown_from_52w_high_pct` 0.77)
- **Grounding:** 19/19 evidence items verified (100%) - fully grounded
- **Documents shown to the explainer:** `SYN-TR-MARM-20260812`, `SYN-NW-MARM-20260915-RATING`, `SYN-NW-MARM-20260812-MOVE`, `SYN-NW-MARM-20260812-PR`

#### Market narrative vs variant view

**Market narrative.** With a -28.85% drawdown from the 52-week high and an RSI of 28.48, and short interest at 19.63% of float (81.54% change over one month), the price implies the market expects the problem behind the last report to persist.

**Variant view.** The evidence points to a one-off rather than a broken business: narrative signals transitory_language x11, no_share_loss x1, no_demand_weakness x1, no_customer_churn x1, no_pricing_pressure x1, demand_resilience x5, backlog_strength x3; a -5.736% revision to NTM EPS consensus over 3 months is a trim, not a collapse.

#### Why the dislocation exists

Holders of a former uptrend sold the miss on heavy volume and trailing-growth screens extrapolated it. If the next report shows the one-off reversing, the gap can close; if the issue recurs, the market was right.

#### Quantitative evidence

| Check | Feature | Value | Interpretation |
|---|---|---:|---|
| ✓ verified | `market_cap_usd_bn` | 4.28 | screen condition market_cap_usd_bn between 2 and 20: passes; sector median 5.581 |
| ✓ verified | `sma_50_vs_sma_200_pct` | 6.924 | screen condition sma_50_vs_sma_200_pct > 0: passes; sector median 8.145 |
| ✓ verified | `return_12m_ex_1m_pct` | 19.24 | screen condition return_12m_ex_1m_pct > 0: passes; sector median -1.729 |
| ✓ verified | `drawdown_from_52w_high_pct` | -28.85 | screen condition drawdown_from_52w_high_pct between -40 and -15: passes; sector median -20.81 |
| ✓ verified | `max_volume_ratio_20d` | 3.116 | screen condition max_volume_ratio_20d >= 2: passes; sector median 1.716 |
| ✓ verified | `rsi_14` | 28.48 | screen condition rsi_14 < 40: passes; sector median 52.62 |
| ✓ verified | `fcf_yield_pct` | 7.908 | screen condition fcf_yield_pct > 4: passes; sector median 2.719 |
| ✓ verified | `revenue_growth_yoy_pct` | 13.7 | screen condition revenue_growth_yoy_pct > 8: passes; sector median 11.3 |
| ✓ verified | `short_interest_pct_float` | 19.63 | screen condition short_interest_pct_float > 6: passes; sector median 2.8 |
| ✓ verified | `revenue_growth_last_q_yoy_pct` | 7.389 | latest quarter revenue still growing year on year; sector median 9.201 |
| ✓ verified | `eps_revision_3m_pct` | -5.736 | modest trim to next-12m EPS consensus over 3 months |
| ✓ verified | `revenue_revision_3m_pct` | -4.991 | revenue consensus trimmed |
| ✓ verified | `operating_margin_change_yoy_pp` | -0.3818 | operating margin broadly stable year on year; sector median -0.06139 |
| ✓ verified | `last_eps_surprise_pct` | -0.9134 | last quarter missed EPS consensus |

#### Narrative evidence

> ✓ “Revenue of $611.6 million grew 7.4% year over year, but that figure absorbs a headwind of approximately $43.2 million from the temporary supply-chain disruption.”
>
> — Mei Abernathy · `SYN-TR-MARM-20260812` · grounding: ✓ verified
>
> _the hit is framed as one-off or temporary (transcript, Mei Abernathy; tag transitory_language)_

> ✓ “In the first 6 weeks of the third quarter, orders are running up 11% year over year.”
>
> — Mei Abernathy · `SYN-TR-MARM-20260812` · grounding: ✓ verified
>
> _underlying demand is described as intact (transcript, Mei Abernathy; tag demand_resilience)_

> ✓ “Our order book ended the quarter at $1.30 billion, up 11.1% year over year, and book-to-bill was 1.06.”
>
> — Mei Abernathy · `SYN-TR-MARM-20260812` · grounding: ✓ verified
>
> _backlog or orders are described as strong (transcript, Mei Abernathy; tag backlog_strength)_

> ✓ “Price realization was positive 2.0% in the quarter, and we did not lose a single top-20 customer.”
>
> — Mei Abernathy · `SYN-TR-MARM-20260812` · grounding: ✓ verified
>
> _management says customers are not leaving (transcript, Mei Abernathy; tag no_customer_churn)_

> ✓ “Gross margin was 47.4%, down 53 basis points from a year ago, reflecting lower absorption on the reduced volume, which we expect to reverse as volumes normalize.”
>
> — Hannah Sterling · `SYN-TR-MARM-20260812` · grounding: ✓ verified
>
> _Cuts against the thesis: margins are under pressure (transcript, Hannah Sterling; tag margin_pressure)_

#### Catalysts

- Next earnings report in about 35 days (days_to_next_earnings).
- Next quarter's results showing the one-off item reversing, as management said it would.
- Share repurchases or dividends that management cited.

#### Risks

- Bearish signals in the documents: margin_pressure x2, results_miss x3.
- Short interest at 19.63% of float can extend the decline if the next print disappoints.
- The issue management calls temporary or limited proves larger or recurring.

#### Invalidation triggers

- revenue_growth_last_q_yoy_pct turns negative in the next report.
- eps_revision_3m_pct falls to -15 or below.
- Management lowers or withdraws guidance, or stops describing the issue as temporary.
- Price makes new lows below the current drawdown of -28.85% while short interest keeps rising.

#### Data gaps

- Heuristic baseline: dislocation type, conviction and text come from transparent rules over feature thresholds and phrase-matched narrative signals (aitrading.agent.offline), not from an analyst's or an LLM's judgement; use it as a starting point.
- Decision trace: transitory_fundamental_shock scored 9.5 (narrative 8, quant 1.5); runner-up structural_decline_value_trap scored 2.5

**Grounding score:** 19/19 evidence items verified (100%) - fully grounded.

### 5.3 #3 GOLF - Goldcrest Facility Services

**Thesis:** Goldcrest Facility Services (GOLF) looks like a transitory shock: the shares show a -21.01% drawdown from the 52-week high and an RSI of 35.28, yet management frames the hit as temporary and latest-quarter revenue growth of 11.19% (18.43% trailing) is still positive.

- **Dislocation type:** Transitory fundamental shock (`transitory_fundamental_shock`)
- **Actionable:** yes · **Conviction:** high
- **Rank score:** 0.667 (`fcf_yield_pct` 0.85, `revenue_growth_yoy_pct` 1.00, `drawdown_from_52w_high_pct` 0.15)
- **Grounding:** 19/19 evidence items verified (100%) - fully grounded
- **Documents shown to the explainer:** `SYN-TR-GOLF-20260807`, `SYN-NW-GOLF-20260917-RATING`, `SYN-NW-GOLF-20260807-MOVE`, `SYN-NW-GOLF-20260807-PR`

#### Market narrative vs variant view

**Market narrative.** With a -21.01% drawdown from the 52-week high and an RSI of 35.28, and short interest at 11.58% of float (52.04% change over one month), the price implies the market expects the problem behind the last report to persist.

**Variant view.** The evidence points to a one-off rather than a broken business: narrative signals transitory_language x12, no_share_loss x1, no_customer_churn x1, no_pricing_pressure x1, demand_resilience x7, backlog_strength x3; a -4.773% revision to NTM EPS consensus over 3 months is a trim, not a collapse.

#### Why the dislocation exists

Holders of a former uptrend sold the miss on heavy volume and trailing-growth screens extrapolated it. If the next report shows the one-off reversing, the gap can close; if the issue recurs, the market was right.

#### Quantitative evidence

| Check | Feature | Value | Interpretation |
|---|---|---:|---|
| ✓ verified | `market_cap_usd_bn` | 6.68 | screen condition market_cap_usd_bn between 2 and 20: passes; sector median 6.511 |
| ✓ verified | `sma_50_vs_sma_200_pct` | 12.73 | screen condition sma_50_vs_sma_200_pct > 0: passes; sector median 3.463 |
| ✓ verified | `return_12m_ex_1m_pct` | 47.56 | screen condition return_12m_ex_1m_pct > 0: passes; sector median -4.409 |
| ✓ verified | `drawdown_from_52w_high_pct` | -21.01 | screen condition drawdown_from_52w_high_pct between -40 and -15: passes; sector median -21.2 |
| ✓ verified | `max_volume_ratio_20d` | 2.815 | screen condition max_volume_ratio_20d >= 2: passes; sector median 1.979 |
| ✓ verified | `rsi_14` | 35.28 | screen condition rsi_14 < 40: passes; sector median 61.2 |
| ✓ verified | `fcf_yield_pct` | 8.017 | screen condition fcf_yield_pct > 4: passes; sector median 5.32 |
| ✓ verified | `revenue_growth_yoy_pct` | 18.43 | screen condition revenue_growth_yoy_pct > 8: passes; sector median 7.716 |
| ✓ verified | `short_interest_pct_float` | 11.58 | screen condition short_interest_pct_float > 6: passes; sector median 3.584 |
| ✓ verified | `revenue_growth_last_q_yoy_pct` | 11.19 | latest quarter revenue still growing year on year; sector median 8.502 |
| ✓ verified | `eps_revision_3m_pct` | -4.773 | modest trim to next-12m EPS consensus over 3 months |
| ✓ verified | `revenue_revision_3m_pct` | -3.436 | revenue consensus trimmed |
| ✓ verified | `operating_margin_change_yoy_pp` | 0.2922 | operating margin broadly stable year on year; sector median -0.1626 |
| ✓ verified | `last_eps_surprise_pct` | -2.51 | last quarter missed EPS consensus |

#### Narrative evidence

> ✓ “Revenue of $744.1 million grew 11.2% year over year, but that figure absorbs a headwind of approximately $55.4 million from the one-time ERP cut-over.”
>
> — Alicia Montague · `SYN-TR-GOLF-20260807` · grounding: ✓ verified
>
> _the hit is framed as one-off or temporary (transcript, Alicia Montague; tag transitory_language)_

> ✓ “Order intake was unaffected and actually grew 7% in the quarter; the issue was entirely on the shipping side.”
>
> — Alicia Montague · `SYN-TR-GOLF-20260807` · grounding: ✓ verified
>
> _underlying demand is described as intact (transcript, Alicia Montague; tag demand_resilience)_

> ✓ “Our contracted backlog ended the quarter at $1.71 billion, up 7.8% year over year, and book-to-bill was 1.05.”
>
> — Alicia Montague · `SYN-TR-GOLF-20260807` · grounding: ✓ verified
>
> _backlog or orders are described as strong (transcript, Alicia Montague; tag backlog_strength)_

> ✓ “Price realization was positive 2.0% in the quarter, and we did not lose a single top-10 customer.”
>
> — Alicia Montague · `SYN-TR-GOLF-20260807` · grounding: ✓ verified
>
> _management says customers are not leaving (transcript, Alicia Montague; tag no_customer_churn)_

> ✓ “Results were affected by the one-time ERP cut-over, which reduced revenue by approximately $55.4 million; orders and contracted backlog grew year over year.”
>
> — `SYN-NW-GOLF-20260807-PR` · grounding: ✓ verified
>
> _Cuts against the thesis: demand is described as weak (news; tag demand_weakness)_

#### Catalysts

- Next earnings report in about 34 days (days_to_next_earnings).
- Next quarter's results showing the one-off item reversing, as management said it would.
- Share repurchases or dividends that management cited.

#### Risks

- Bearish signals in the documents: demand_weakness x1, margin_pressure x2, results_miss x3.
- Short interest at 11.58% of float can extend the decline if the next print disappoints.
- The issue management calls temporary or limited proves larger or recurring.

#### Invalidation triggers

- revenue_growth_last_q_yoy_pct turns negative in the next report.
- eps_revision_3m_pct falls to -15 or below.
- Management lowers or withdraws guidance, or stops describing the issue as temporary.
- Price makes new lows below the current drawdown of -21.01% while short interest keeps rising.

#### Data gaps

- Heuristic baseline: dislocation type, conviction and text come from transparent rules over feature thresholds and phrase-matched narrative signals (aitrading.agent.offline), not from an analyst's or an LLM's judgement; use it as a starting point.
- Decision trace: transitory_fundamental_shock scored 9 (narrative 7.5, quant 1.5); runner-up structural_decline_value_trap scored 3

**Grounding score:** 19/19 evidence items verified (100%) - fully grounded.

### 5.4 #4 STON - Stonebridge Diagnostics

**Thesis:** Stonebridge Diagnostics (STON) looks like a transitory shock: the shares show a -20.02% drawdown from the 52-week high and an RSI of 31.82, yet management frames the hit as temporary and latest-quarter revenue growth of 6.77% (15.02% trailing) is still positive.

- **Dislocation type:** Transitory fundamental shock (`transitory_fundamental_shock`)
- **Actionable:** yes · **Conviction:** high
- **Rank score:** 0.590 (`fcf_yield_pct` 0.92, `revenue_growth_yoy_pct` 0.85, `drawdown_from_52w_high_pct` 0.00)
- **Grounding:** 19/19 evidence items verified (100%) - fully grounded
- **Documents shown to the explainer:** `SYN-TR-STON-20260812`, `SYN-NW-STON-20260922-RATING`, `SYN-NW-STON-20260812-MOVE`, `SYN-NW-STON-20260812-PR`

#### Market narrative vs variant view

**Market narrative.** With a -20.02% drawdown from the 52-week high and an RSI of 31.82, and short interest at 13.92% of float (60.83% change over one month), the price implies the market expects the problem behind the last report to persist.

**Variant view.** The evidence points to a one-off rather than a broken business: narrative signals transitory_language x14, no_share_loss x1, no_customer_churn x1, no_pricing_pressure x1, demand_resilience x8, backlog_strength x3, fx_headwind x8; a -7.137% revision to NTM EPS consensus over 3 months is a trim, not a collapse.

#### Why the dislocation exists

Holders of a former uptrend sold the miss on heavy volume and trailing-growth screens extrapolated it. If the next report shows the one-off reversing, the gap can close; if the issue recurs, the market was right.

#### Quantitative evidence

| Check | Feature | Value | Interpretation |
|---|---|---:|---|
| ✓ verified | `market_cap_usd_bn` | 3.371 | screen condition market_cap_usd_bn between 2 and 20: passes; sector median 5.581 |
| ✓ verified | `sma_50_vs_sma_200_pct` | 11.71 | screen condition sma_50_vs_sma_200_pct > 0: passes; sector median 8.145 |
| ✓ verified | `return_12m_ex_1m_pct` | 30.47 | screen condition return_12m_ex_1m_pct > 0: passes; sector median -1.729 |
| ✓ verified | `drawdown_from_52w_high_pct` | -20.02 | screen condition drawdown_from_52w_high_pct between -40 and -15: passes; sector median -20.81 |
| ✓ verified | `max_volume_ratio_20d` | 3.752 | screen condition max_volume_ratio_20d >= 2: passes; sector median 1.716 |
| ✓ verified | `rsi_14` | 31.82 | screen condition rsi_14 < 40: passes; sector median 52.62 |
| ✓ verified | `fcf_yield_pct` | 8.201 | screen condition fcf_yield_pct > 4: passes; sector median 2.719 |
| ✓ verified | `revenue_growth_yoy_pct` | 15.02 | screen condition revenue_growth_yoy_pct > 8: passes; sector median 11.3 |
| ✓ verified | `short_interest_pct_float` | 13.92 | screen condition short_interest_pct_float > 6: passes; sector median 2.8 |
| ✓ verified | `revenue_growth_last_q_yoy_pct` | 6.77 | latest quarter revenue still growing year on year; sector median 9.201 |
| ✓ verified | `eps_revision_3m_pct` | -7.137 | modest trim to next-12m EPS consensus over 3 months |
| ✓ verified | `revenue_revision_3m_pct` | -3.325 | revenue consensus trimmed |
| ✓ verified | `operating_margin_change_yoy_pp` | -0.1675 | operating margin broadly stable year on year; sector median -0.06139 |
| ✓ verified | `last_eps_surprise_pct` | -0.6533 | last quarter missed EPS consensus |

#### Narrative evidence

> ✓ “Revenue of $383.7 million grew 6.8% year over year, but that figure absorbs a headwind of approximately $34.9 million from the one-time currency translation and hedge-settlement impact.”
>
> — Nathaniel Jansen · `SYN-TR-STON-20260812` · grounding: ✓ verified
>
> _the hit is framed as one-off or temporary (transcript, Nathaniel Jansen; tag transitory_language)_

> ✓ “On a constant-currency basis our Northern European business grew 11% and order intake there was a record.”
>
> — Nathaniel Jansen · `SYN-TR-STON-20260812` · grounding: ✓ verified
>
> _underlying demand is described as intact (transcript, Nathaniel Jansen; tag demand_resilience)_

> ✓ “What happened is specific and identifiable: a one-time currency impact tied to the abrupt devaluation of the Brazilian real, which hit translated revenue and forced an early settlement of our hedge book.”
>
> — Nathaniel Jansen · `SYN-TR-STON-20260812` · grounding: ✓ verified
>
> _currency is cited as a headwind (transcript, Nathaniel Jansen; tag fx_headwind)_

> ✓ “Our backlog ended the quarter at $720.9 million, up 8.8% year over year, and book-to-bill was 1.13.”
>
> — Nathaniel Jansen · `SYN-TR-STON-20260812` · grounding: ✓ verified
>
> _backlog or orders are described as strong (transcript, Nathaniel Jansen; tag backlog_strength)_

> ✓ “Gross margin was 40.6%, down 39 basis points from a year ago, reflecting lower absorption on the reduced volume, which we expect to reverse as volumes normalize.”
>
> — Elena Underwood · `SYN-TR-STON-20260812` · grounding: ✓ verified
>
> _Cuts against the thesis: margins are under pressure (transcript, Elena Underwood; tag margin_pressure)_

#### Catalysts

- Next earnings report in about 29 days (days_to_next_earnings).
- Next quarter's results showing the one-off item reversing, as management said it would.
- Share repurchases or dividends that management cited.

#### Risks

- Bearish signals in the documents: margin_pressure x2, results_miss x3.
- Short interest at 13.92% of float can extend the decline if the next print disappoints.
- The issue management calls temporary or limited proves larger or recurring.

#### Invalidation triggers

- revenue_growth_last_q_yoy_pct turns negative in the next report.
- eps_revision_3m_pct falls to -15 or below.
- Management lowers or withdraws guidance, or stops describing the issue as temporary.
- Price makes new lows below the current drawdown of -20.02% while short interest keeps rising.

#### Data gaps

- Heuristic baseline: dislocation type, conviction and text come from transparent rules over feature thresholds and phrase-matched narrative signals (aitrading.agent.offline), not from an analyst's or an LLM's judgement; use it as a starting point.
- Decision trace: transitory_fundamental_shock scored 10.5 (narrative 9, quant 1.5); runner-up structural_decline_value_trap scored 2.5

**Grounding score:** 19/19 evidence items verified (100%) - fully grounded.

### 5.5 #5 THOI - Thornbury Industries

**Thesis:** Thornbury Industries (THOI) looks like a value trap: its 7.612% trailing FCF yield sits on latest-quarter revenue growth of -5.529% (10.55% trailing) and a -26.47% revision to NTM EPS consensus over 3 months.

- **Dislocation type:** Structural decline value trap (`structural_decline_value_trap`)
- **Actionable:** no · **Conviction:** high
- **Rank score:** 0.590 (`fcf_yield_pct` 0.62, `revenue_growth_yoy_pct` 0.23, `drawdown_from_52w_high_pct` 0.92)
- **Grounding:** 19/19 evidence items verified (100%) - fully grounded
- **Documents shown to the explainer:** `SYN-TR-THOI-20260814`, `SYN-NW-THOI-20260911-RATING`, `SYN-NW-THOI-20260814-MOVE`, `SYN-NW-THOI-20260814-PR`

#### Market narrative vs variant view

**Market narrative.** With a -31.06% drawdown from the 52-week high and an RSI of 29.2, and short interest at 10.87% of float (51.32% change over one month), the price implies the market expects a lasting deterioration in the business.

**Variant view.** The evidence agrees with the market. Bearish quant flags: revenue_growth_last_q_yoy_pct, eps_revision_3m_pct, revenue_revision_3m_pct; bearish narrative signals: structural_concern x4, pricing_pressure x1, guidance_withdrawn x1, evasive_answer x7, demand_weakness x2, margin_pressure x3, results_miss x2.

#### Why the dislocation exists

Trailing metrics lag: the screen sees a high trailing FCF yield and TTM growth, while the latest quarter, estimate revisions and management commentary point down. The low price reflects falling forward numbers, so the gap is unlikely to close until revisions stop.

#### Quantitative evidence

| Check | Feature | Value | Interpretation |
|---|---|---:|---|
| ✓ verified | `market_cap_usd_bn` | 6.688 | screen condition market_cap_usd_bn between 2 and 20: passes; sector median 6.511 |
| ✓ verified | `sma_50_vs_sma_200_pct` | 3.849 | screen condition sma_50_vs_sma_200_pct > 0: passes; sector median 3.463 |
| ✓ verified | `return_12m_ex_1m_pct` | 23.34 | screen condition return_12m_ex_1m_pct > 0: passes; sector median -4.409 |
| ✓ verified | `drawdown_from_52w_high_pct` | -31.06 | screen condition drawdown_from_52w_high_pct between -40 and -15: passes; sector median -21.2 |
| ✓ verified | `max_volume_ratio_20d` | 3.658 | screen condition max_volume_ratio_20d >= 2: passes; sector median 1.979 |
| ✓ verified | `rsi_14` | 29.2 | screen condition rsi_14 < 40: passes; sector median 61.2 |
| ✓ verified | `fcf_yield_pct` | 7.612 | screen condition fcf_yield_pct > 4: passes; sector median 5.32 |
| ✓ verified | `revenue_growth_yoy_pct` | 10.55 | screen condition revenue_growth_yoy_pct > 8: passes; sector median 7.716 |
| ✓ verified | `short_interest_pct_float` | 10.87 | screen condition short_interest_pct_float > 6: passes; sector median 3.584 |
| ✓ verified | `revenue_growth_last_q_yoy_pct` | -5.529 | latest quarter revenue is shrinking year on year: trailing growth overstates the current trend; sector median 8.502 |
| ✓ verified | `eps_revision_3m_pct` | -26.47 | hard cut to next-12m EPS consensus over 3 months |
| ✓ verified | `revenue_revision_3m_pct` | -10.3 | revenue consensus cut sharply |
| ✓ verified | `operating_margin_change_yoy_pp` | -0.6818 | operating margin broadly stable year on year; sector median -0.1626 |
| ✓ verified | `last_eps_surprise_pct` | -4.877 | last quarter missed EPS consensus |

#### Narrative evidence

> ✓ “Competitive intensity increased in the quarter.”
>
> — Sofia Petrova · `SYN-TR-THOI-20260814` · grounding: ✓ verified
>
> _a structural or competitive deterioration is acknowledged (transcript, Sofia Petrova; tag structural_concern)_

> ✓ “Given the uncertainty, we are no longer reaffirming the medium-term financial framework we outlined at our last investor day.”
>
> — Sofia Petrova · `SYN-TR-THOI-20260814` · grounding: ✓ verified
>
> _guidance was withdrawn or no longer reaffirmed (transcript, Sofia Petrova; tag guidance_withdrawn)_

> ✓ “The year-over-year change reflected lower volumes in our core product lines and lower average selling prices, partially offset by growth in services.”
>
> — Arjun Yamamoto · `SYN-TR-THOI-20260814` · grounding: ✓ verified
>
> _pricing pressure is acknowledged (transcript, Arjun Yamamoto; tag pricing_pressure)_

> ✓ “We will provide an update when we have better visibility.”
>
> — Sofia Petrova · `SYN-TR-THOI-20260814` · grounding: ✓ verified
>
> _management deflects or declines to quantify (transcript, Sofia Petrova; tag evasive_answer)_

> ✓ “The outlook reflects the competitive and pricing dynamics we have discussed, and we have taken what we believe is a prudent view of the second half.”
>
> — Arjun Yamamoto · `SYN-TR-THOI-20260814` · grounding: ✓ verified
>
> _Cuts against the thesis: guidance is described as deliberately conservative (transcript, Arjun Yamamoto; tag guidance_conservative)_

#### Catalysts

- Next earnings report in about 40 days (days_to_next_earnings).
- None identified for a re-rating: estimate revisions would first need to stop falling.

#### Risks

- Short interest at 10.87% of float can extend the decline if the next print disappoints.
- Low expectations and a crowded short can cause sharp rallies even if the decline is structural.

#### Invalidation triggers

- Latest-quarter revenue growth returns to the trailing rate (10.55%) with eps_revision_3m_pct turning positive.
- Management quantifies the problem and shows it reversing (margins and orders stabilising).
- Price makes new lows below the current drawdown of -31.06% while short interest keeps rising.

#### Data gaps

- Heuristic baseline: dislocation type, conviction and text come from transparent rules over feature thresholds and phrase-matched narrative signals (aitrading.agent.offline), not from an analyst's or an LLM's judgement; use it as a starting point.
- Decision trace: structural_decline_value_trap scored 16 (narrative 10, quant 6); runner-up guidance_reset_overreaction scored 2

**Grounding score:** 19/19 evidence items verified (100%) - fully grounded.

Not explained (ranked below the explanation cut-off): #6 INGD, #7 IVOC, #8 BLAC, #9 JESS, #10 VELB.

## Appendix A. Audit trail

### A.1 LLM calls

_No LLM calls were made (LLM: none (offline heuristics))._

### A.2 Vendor push-down query

_Not used: every condition was evaluated locally by the deterministic screen engine._

### A.3 Warnings

1. DLCR: withheld SYN-FL-DLCR-20260812-10Q: over max_documents_per_ticker=4
2. DLCR: withheld SYN-FL-DLCR-20260803-8K: over max_documents_per_ticker=4
3. MARM: withheld SYN-FL-MARM-20260821-10Q: over max_documents_per_ticker=4
4. MARM: withheld SYN-FL-MARM-20260812-8K: over max_documents_per_ticker=4
5. GOLF: withheld SYN-FL-GOLF-20260814-10Q: over max_documents_per_ticker=4
6. GOLF: withheld SYN-FL-GOLF-20260807-8K: over max_documents_per_ticker=4
7. STON: withheld SYN-NW-STON-20260701-CORP: over max_documents_per_ticker=4
8. STON: withheld SYN-FL-STON-20260817-10Q: over max_documents_per_ticker=4
9. STON: withheld SYN-FL-STON-20260812-8K: over max_documents_per_ticker=4
10. THOI: withheld SYN-FL-THOI-20260826-10Q: over max_documents_per_ticker=4
11. THOI: withheld SYN-FL-THOI-20260814-8K: over max_documents_per_ticker=4
