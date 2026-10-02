# AItrading

An end-to-end equity research platform that turns market data into **defensible** investment ideas,
and research ideas into **tested** strategies:

* **Screen and explain**: describe an observation in plain English. The platform translates it into a
  precise screen, scans the US equity universe, ranks the survivors, reads their earnings releases,
  calls and filings, and explains why each dislocation exists. It also flags value traps.
* **Idea lab**: type an idea ("Fama-French 3-factor model", "12-1 momentum deciles", "buy SPY above
  its 200-day"). The platform gathers point-in-time data, builds the strategy, backtests it with costs,
  attributes it to academic factors and gives a skeptical verdict.
* **Auto strategy creator**: finds ideas in research papers and on the web, verifies what they claim,
  asks you idea by idea whether to try them, then backtests and stress-tests the ones you accept.
* **Simulated trading**: save a strategy and run it forward as a paper portfolio, using the same code
  path as the backtest. No broker is involved.

Claude (`claude-opus-5-5`) does the reasoning: translating intent, explaining, interpreting, reading
papers. **It never computes the numbers that select stocks or score a backtest.** Those come from a
deterministic engine. Every quote Claude cites is checked verbatim against its source, and every
number against the data, before it reaches a report.

> Research tool. Results are simulated and not investment advice.

---

## Quick start (your PC)

Full guide for Windows / macOS / Linux: **[docs/QUICKSTART_PC.md](docs/QUICKSTART_PC.md)**.

```bash
git clone https://github.com/tiangangzh/AItrading.git && cd AItrading
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[free]"

export SEC_USER_AGENT="Your Name you@example.com"         # required by the SEC for EDGAR access
export ANTHROPIC_API_KEY="sk-ant-..."                     # optional: lets Claude do the reasoning

aitrading demo                                            # offline self-test (no internet, no keys)
aitrading run "US mid-caps in uptrends that pulled back 15-40% on heavy volume, RSI under 40, FCF yield above 4%"
aitrading backtest "Fama-French 3 factor model"
aitrading discover
aitrading dashboard
```

Without an Anthropic key everything still runs, using built-in rule-based engines instead of Claude
(the CLI tells you when it does this).

---

## What you can do

### 1. Screen the market and explain the dislocations: `aitrading run`

```bash
aitrading run "Find US mid-caps ($2-20B) in established uptrends (50-day above 200-day, positive 12-1 momentum) \
that pulled back 15-40% from their 52-week highs on heavy volume, RSI under 40, FCF yield above 4%, revenue growth \
above 8%, short interest above 6% of float. Rank by FCF yield, growth and drawdown, then explain the dislocation."
```

1. **NL quantitative engine.** The observation becomes a typed `ScreenSpec` over an
   80+-feature catalog. Inspect it with `aitrading spec "..."`. Anything the catalog cannot express is
   listed, not silently dropped.
2. **Technical scanner**: price, volume, RSI, MACD, moving-average structure, drawdowns, relative
   strength, volatility, options (IV rank, put/call).
3. **Fundamental scanner**: FCF yield, EV multiples, growth, margins and their trends, estimate
   revisions, balance sheet, short interest.
4. **Screen and rank.** The report shows a funnel of how many names each condition removed, plus
   feature coverage and a percentile-rank composite.
5. **Narrative engine**: earnings-call transcripts, news, filings and research, within each data
   vendor's licensing boundary.
6. **Explanation agent.** For each top name it gives what the market is pricing in, the variant view,
   why the dislocation exists, catalysts, risks, invalidation triggers and conviction. It also says
   whether this is a genuine dislocation or a value trap. Each quote and number carries a ✓/✗
   grounding mark.

Output: `aitrading_output/<run>/report.html` (opened automatically), `report.md`, `result.json`,
`features.csv` and `documents.json`.

### 2. Backtest an idea: `aitrading backtest`

```bash
aitrading library                                   # ~25 classic ideas with academic references
aitrading backtest "Fama-French 3 factor model"     # builds SMB/HML from your universe, compares with Ken French's data
aitrading backtest "12-1 momentum, deciles, long-short, monthly" --save momentum
aitrading backtest "quality at a reasonable price, long only top quintile" --start 2015-01-01 --costs-bps 15
aitrading backtest "hold SPY when it is above its 200-day moving average"
```

The report covers:
- the idea in plain English;
- growth of $1 against the benchmark, and drawdowns;
- CAGR, Sharpe and Sortino, with Newey-West t-stats;
- returns by quintile and the information coefficient;
- alpha and betas versus CAPM, FF3, Carhart or FF5, using the official Kenneth French factors;
- turnover and costs;
- current holdings, data provenance, and a skeptical verdict (robust / promising / weak / likely
  spurious / inconclusive) whose cited numbers are machine-checked.

Point-in-time discipline:
- Signals use only data public at each rebalance date: fundamentals by filing date.
- Trades execute at the next session's close.
- Delistings are handled.
- Survivorship bias of today's universe is called out in every report.

### 3. Run a strategy as simulated trading: `aitrading strategy`

```bash
aitrading strategy list
aitrading strategy run momentum        # today's target portfolio -> simulated orders -> ledger
aitrading strategy status momentum     # paper NAV vs what the backtest predicted
```

Schedule `aitrading strategy run <name>` daily (Task Scheduler / cron). It only trades on the
strategy's rebalance days and is idempotent. Each strategy gets a page showing the idea, holdings,
trade blotter and simulated-vs-backtest performance.

### 4. Let it find ideas for you: `aitrading discover`

```bash
aitrading discover                                  # arXiv q-fin + your feeds + Claude web search
aitrading discover --query "post-earnings announcement drift"
aitrading discover --url https://arxiv.org/abs/XXXX.XXXXX
aitrading discover --pdf paper.pdf
aitrading ideas list                                # the idea inbox
aitrading ideas try <id>
```

For each idea you see:
- what it claims, and the reported Sharpe / t-stat;
- verbatim evidence quotes, each checked against the source;
- whether your data can test it, and the closest classic strategy.

Then it asks **"Try this strategy?"**. On yes it backtests the idea and runs a replication suite:
- the two halves of the sample;
- post-publication decay;
- 0 / 25 bps costs;
- nearby parameters;
- replicated vs claimed Sharpe.

The verdict is one of: replicates, partially replicates, fails to replicate, or inconclusive. You can
then save the strategy for simulated trading.

Web and paper text is treated as untrusted data: instruction-like passages are flagged and need your
confirmation, and URLs Claude "finds" must appear in actual search results.

### 5. Everything in one place: `aitrading dashboard`

An index page linking every screen, backtest, strategy and idea.

---

## Data

| `--provider` | Data | Needs |
|---|---|---|
| `free` (default) | Yahoo Finance via `yfinance` (prices, short interest, estimates, options snapshot) + SEC EDGAR (point-in-time fundamentals, 8-K earnings releases, 10-Q/10-K MD&A) | `pip install -e ".[free]"`, `SEC_USER_AGENT` |
| `synthetic` | Built-in simulated US market with planted dislocations and value traps, transcripts and news (for tests and the demo) | nothing |
| `bloomberg` | BQL inside BQuant (screen push-down) or the Desktop API (`blpapi`) | Terminal / BQuant entitlement |
| `lseg` | LSEG Data Library (`SCREEN()` push-down, history, news, filings) | Workspace or platform session |
| `capiq` | S&P Capital IQ GDS API (+ Kensho transcripts if licensed) | Capital IQ API entitlement |

Vendor field codes live in editable field maps (`src/aitrading/data/fieldmaps/*.json`), each tagged
confirmed / corrected / unverifiable from the research in
[docs/VENDOR_REFERENCE.md](docs/VENDOR_REFERENCE.md). Only verified fields are pushed down into
vendor screens, and every predicate is re-checked locally.

Limits of the free edition:
- **No transcripts.** It uses earnings releases and MD&A instead.
- **Snapshots only for some data.** Short interest, estimates and options are current snapshots, so
  they are only used for today's date.
- **Small universe.** It screens the tickers you give it (default ~150).

## Architecture: where the reasoning sits

Should the reasoning layer sit inside the Bloomberg Terminal (ASKB / BQuant), or on top of it, via
LSEG / S&P into a frontier model? That decision is recorded in
**[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**: three designs, scored by independent judges on
governance, quantitative correctness and engineering.

**Decision.** Claude sits on top, compute is pushed down into each vendor, and a deny-by-default
licensing boundary is enforced in code. Claude translates and explains; numbers are computed where
the data is licensed to live. Rollout is phased behind written vendor and legal gates.

```
 observation / idea / paper
          │
          ▼
 Claude: NL -> typed ScreenSpec / StrategySpec          (validated against the feature catalog)
          │
          ▼
 deterministic engine: vendor push-down (BQL / SCREEN) + local re-check,
 point-in-time features, screen, rank, backtest, factor regressions
          │
          ▼
 narrative engine (licensing boundary decides what text may reach the LLM)
          │
          ▼
 Claude: explanation / interpretation  ──►  programmatic grounding checks  ──►  report + audit trail
```

How Claude is called:
- Structured outputs.
- Adaptive thinking, with an explicit effort setting.
- Server-side refusal fallback: if Claude declines a request, the API retries it on a fallback model.
- A cached, byte-stable system prompt.
- Every call logged with model, request id, tokens and stop reason.

## Project layout

```
src/aitrading/
  core/        domain models, canonical data fields, licensing boundary
  data/        providers: free (Yahoo + SEC), synthetic, bloomberg, lseg, capiq; field maps; cache
  screen/      feature catalog, ScreenSpec, NL translator, feature engine, screen engine, BQL/LSEG compilers
  technical/ fundamental/ positioning/   feature computation
  rank/        cross-sectional ranking
  narrative/   document retrieval, excerpts, signals, quote/number grounding
  agent/       explanation agent (Claude + offline), interactive research agent (tool runner)
  strategy/    StrategySpec, idea library, idea translator, results interpreter
  backtest/    engine, metrics, regressions, quantile/IC analysis, runner
  factors/     Kenneth French data loader, Fama-French factor construction
  discovery/   idea sources (arXiv, feeds, web search, URL/PDF), extraction, ranking, inbox, replication
  trading/     strategy store, paper-trading ledger
  report/      Markdown/HTML reports, SVG charts, dashboard
  pipeline.py idealab.py cli.py cli_lab.py
docs/          ARCHITECTURE.md (ADR), VENDOR_REFERENCE.md, QUICKSTART_PC.md
tests/         offline test suite (no network, no API key)
```

## Development

```bash
pip install -e ".[dev,free]"
python -m pytest -q          # fully offline: fixtures, fakes and the synthetic market
```

## Honest limitations

* **Backtests on free data have survivorship bias.** The universe is today's tickers. Reports say so;
  use point-in-time constituents from an institutional provider for publishable results.
* **The synthetic market tests the pipeline, not real-world skill.** The offline explainer is
  rule-based, and `aitrading demo` scores it against the synthetic market's planted ground truth.
* **Vendor adapters are untested against live accounts.** They are built from verified documentation
  and tested against fakes. Run each adapter's `verify_fields()` on your entitlement before relying on
  a field.
* **No broker connectivity.** Trading is simulated only.
