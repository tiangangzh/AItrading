# Run it on your PC

AItrading is a normal Python program: clone it from GitHub, install it, and run it from a terminal
on Windows, macOS or Linux. On a PC it uses **real market data from free sources** by default, and
**Claude** for the reasoning when you give it your Anthropic API key. If you have institutional
entitlements, the same program runs on Bloomberg / LSEG / S&P Capital IQ (see `docs/ARCHITECTURE.md`).

| Piece | On your PC (default) | Institutional |
|---|---|---|
| Market data | Yahoo Finance via `yfinance` (prices, short interest, consensus estimates, options) + SEC EDGAR (fundamentals, 8-K earnings releases, 10-Q/10-K MD&A) | Bloomberg (BQL/BQuant), LSEG Data Library, S&P Capital IQ |
| Reasoning | Claude (`claude-opus-5-5`) with your `ANTHROPIC_API_KEY`; built-in rule-based fallback without one | same |
| Universe | ~150 liquid US stocks bundled (mid-cap tilted), or your own ticker list | full US equity universe |

Requirements: Windows 10/11, macOS or Linux; Python 3.10+; Git; ~4 GB RAM; internet access.

---

## 1. Get the code and install (once)

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

Or run `scripts\install.ps1` (Windows) / `./scripts/install.sh` (macOS/Linux), which do the same.

## 2. Set your keys (each new terminal, or add them to your user environment variables)

The SEC asks every automated client to identify itself with a name and email. The Anthropic key is
what lets Claude do the reasoning (create one at <https://console.anthropic.com/>).

```powershell
# Windows PowerShell
$env:SEC_USER_AGENT    = "Jane Doe jane@example.com"
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

```bash
# macOS / Linux
export SEC_USER_AGENT="Jane Doe jane@example.com"
export ANTHROPIC_API_KEY="sk-ant-..."
```

To make them permanent on Windows: *Start -> "Edit environment variables for your account"*.
On macOS/Linux add the two `export` lines to `~/.zshrc` or `~/.bashrc`.

## 3. Run

```bash
aitrading run "Find US mid-caps (\$2-20B) that were in established uptrends (50-day above 200-day, positive 12-1 momentum) but have pulled back 15-40% from their 52-week highs on heavy volume, now oversold (RSI under 40), still generating strong free cash flow (FCF yield above 4%) with revenue growth above 8%, and where short interest is elevated (above 6% of float). Rank by FCF yield, growth and the size of the drawdown, then read the latest earnings releases and explain the dislocation."
```

What happens:

1. Claude translates the observation into a typed screen (inspect it with `aitrading spec "..."`).
2. The program downloads prices, fundamentals, short interest, estimates and options for the
   universe, computes ~80 technical / fundamental / positioning features, applies the screen and
   ranks the survivors. (Claude never computes these numbers.)
3. For the top names it reads the latest SEC earnings press releases and 10-Q/10-K MD&A, and Claude
   explains why each dislocation exists - or says it looks like a value trap.
4. Every quote is checked verbatim against the source filing and every number against the feature
   table; the report marks each one ✓ / ✗.
5. The report is written to `./aitrading_output/` (Markdown + HTML) and opened in your browser.

The first run downloads and caches data in `~/.aitrading/cache` (a few minutes for ~150 tickers);
reruns are fast.

### Your own tickers

```bash
aitrading run --tickers CROX,DECK,ELF,ONON,SKX "..."
aitrading run --universe-file my_watchlist.txt "..."     # one ticker per line, or a CSV with a 'ticker' column
```

### Without an Anthropic key

The program still runs end to end with a built-in rule-based translator and explainer (it tells you
when it does this). Add `--offline` to force that mode even when a key is set.

---

## Useful commands

```bash
aitrading run --help                   # all options: --tickers, --top, --explain, --effort, --as-of, --out, --format
aitrading spec "..."                   # show the screen an observation translates to
aitrading screen "..."                 # screen + rank only, no explanations (fast, cheap)
aitrading catalog                      # every screenable feature with units and definitions
aitrading demo                         # offline self-test on a built-in simulated market (no internet, no keys)
```

Cost control with Claude: roughly one call for the screen plus one per explained candidate.
Use `--explain 2` while experimenting and `--effort medium` for cheaper, faster explanations.

## Limits of the free data (vs. the institutional build)

* **Earnings-call transcripts** are not freely licensed. The narrative engine reads the 8-K earnings
  press release and the 10-Q/10-K MD&A instead - good for numbers and management's framing, weaker
  on Q&A tone. With Bloomberg / LSEG / Capital IQ entitlements the transcripts are used.
* **Point-in-time history for snapshots.** Short interest, consensus estimates and options from
  Yahoo are *current* snapshots, so they are used only when `--as-of` is today (the default) or
  within a few days. For historical `--as-of` dates those features are left blank and the report
  says so. Prices and SEC fundamentals *are* point-in-time (a filing is used only after its filing
  date).
* **Universe size.** It screens the tickers you give it (default: the bundled ~150), not the whole
  market. Pass a bigger list for a wider net; download time grows with it.
* Yahoo data via `yfinance` is unofficial and for personal research use; respect Yahoo's terms.

## Troubleshooting

* `SEC_USER_AGENT is not set` - see step 2.
* `No Anthropic credentials found - running offline` - set `ANTHROPIC_API_KEY` (step 2).
* A ticker has blank fundamentals - it may file under a different entity/class, be a foreign
  private issuer (20-F/6-K instead of 10-Q/8-K), or use uncommon XBRL tags. The report lists
  data-coverage warnings.
* Behind a corporate proxy: `pip`, `yfinance` and the SEC client honour `HTTPS_PROXY`.
