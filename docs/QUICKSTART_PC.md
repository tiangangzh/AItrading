# Run it on your PC (small edition)

The full pipeline is built for institutional data (Bloomberg / LSEG / S&P Capital IQ). The small
edition runs the **same pipeline on your own computer**, with three levels you can climb:

| Level | Data | Reasoning | Needs |
|---|---|---|---|
| 1. Demo | Built-in synthetic US market (500 fictional stocks with planted dislocations and value traps, earnings-call transcripts, news) | Offline rule-based translator + explainer | Python only |
| 2. Free real data | Real US stocks: prices, short interest, consensus estimates and options snapshot from Yahoo Finance (via `yfinance`); fundamentals, 8-K earnings press releases and 10-Q/10-K MD&A from SEC EDGAR | Offline rule-based, or Claude (level 3) | Internet + an `SEC_USER_AGENT` string |
| 3. Claude | Either of the above | Claude (`claude-opus-5-5`) translates your observation into a screen and explains each dislocation, with every quote and number machine-checked | Your own `ANTHROPIC_API_KEY` |

Requirements: Windows 10/11, macOS or Linux; Python 3.10 or newer; ~4 GB RAM; ~200 MB disk.

---

## 1. Install (once)

### Windows (PowerShell)

```powershell
git clone https://github.com/tiangangzh/AItrading.git
cd AItrading
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -e ".[free]"
```

### macOS / Linux

```bash
git clone https://github.com/tiangangzh/AItrading.git
cd AItrading
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[free]"
```

Or use the one-shot launchers, which create the virtual environment, install and run the demo:
`scripts\run_demo.ps1 -Free` (Windows) / `./scripts/run_demo.sh --free` (macOS/Linux).

---

## 2. Level 1 - the demo (no keys, no internet after install)

```bash
aitrading demo
```

This runs the representative task end to end on the synthetic market: translate the observation
into a screen, scan ~500 stocks, rank the survivors, read their earnings calls, explain each
dislocation, verify every quote and number, and write the report to `./aitrading_output/`
(Markdown + HTML; the HTML opens in your browser - add `--no-open` to skip that).

The synthetic market knows the right answers (which names are genuine transitory shocks and which
are value traps), so the demo also prints how well the explainer told them apart.

Try your own observation on the synthetic market:

```bash
aitrading run --offline "Large caps above their 200-day with RSI under 35 and FCF yield above 5%"
```

---

## 3. Level 2 - free real data

The SEC asks every automated client to identify itself. Set this once per terminal session
(use your own name and email):

```powershell
# Windows PowerShell
$env:SEC_USER_AGENT = "Jane Doe jane@example.com"
```

```bash
# macOS / Linux
export SEC_USER_AGENT="Jane Doe jane@example.com"
```

Then run against the bundled starter watchlist (~150 liquid US stocks across all sectors, tilted to
mid-caps), or your own tickers:

```bash
aitrading run --provider free --offline "Mid-caps in uptrends that pulled back 15-40% from highs on heavy volume, RSI under 40, FCF yield above 4%, revenue growth above 8%, short interest above 6% of float"

aitrading run --provider free --tickers CROX,DECK,ELF,ONON,SKX --offline "..."
aitrading run --provider free --universe-file my_watchlist.txt --offline "..."
```

The first run downloads and caches data in `~/.aitrading/cache` (a few minutes for ~150 tickers);
reruns are fast.

What the free edition does **not** have, compared with the institutional build:

* **Earnings-call transcripts** are not freely licensed. The narrative engine reads the 8-K earnings
  press release and the 10-Q/10-K MD&A instead - good for numbers and management's framing,
  weaker on Q&A tone.
* **Point-in-time history for snapshots.** Short interest, consensus estimates and options from
  Yahoo are *current* snapshots, so they are only used when `--as-of` is today (or within a few
  days). For historical `--as-of` dates those features are left blank and the report says so.
  Prices and SEC fundamentals *are* point-in-time (filings are used only after their filing date).
* **Universe size.** It screens the tickers you give it, not the whole market. Pass a larger file
  if you want a wider net; expect download time to grow with it.
* Yahoo data via `yfinance` is unofficial and for personal research use; respect Yahoo's terms.

---

## 4. Level 3 - reasoning with Claude

Create an API key at <https://console.anthropic.com/> and set it in your terminal:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."      # Windows PowerShell
```

```bash
export ANTHROPIC_API_KEY="sk-ant-..."      # macOS / Linux
```

Drop `--offline` and Claude takes over the two reasoning steps:

```bash
aitrading run --provider free "Your observation in plain English"
aitrading run "Your observation"            # same, on the synthetic market
```

* Claude translates the observation into a typed screen (you can inspect it with
  `aitrading spec "..."`) and explains each top candidate. It never computes the numbers that select
  stocks; those come from the deterministic engine.
* Every quote must be verbatim from a source document and every number must match the feature
  table; the report marks each one ✓ / ✗.
* Cost: roughly one call for the screen plus one per explained candidate (`--explain 5` by default).
  Use `--explain 2` while experimenting, `--effort medium` for cheaper, faster explanations.

---

## Useful commands

```bash
aitrading catalog                      # every screenable feature with units and definitions
aitrading spec --offline "..."         # show the screen an observation translates to
aitrading screen --offline "..."       # screen + rank only, no explanations
aitrading run --help                   # all options (as-of date, top N, output folder, JSON output)
```

## Troubleshooting

* `SEC_USER_AGENT is not set` - see step 3.
* `No Anthropic credentials found` - set `ANTHROPIC_API_KEY` or add `--offline`.
* A ticker shows blank fundamentals - it may file with the SEC under a different ticker/class, be
  a foreign private issuer (files 20-F/6-K, not 10-Q/8-K), or use uncommon XBRL tags. The report
  lists data-coverage warnings.
* Corporate proxies: `pip` and the data clients honour `HTTPS_PROXY`.
