# Learning from backtests: hypotheses, beliefs and an "intuition" judge

**Question:** can the platform learn from backtest feedback, so that an idea that keeps working gains
credibility, an idea that keeps failing loses it, and new ideas are judged against that experience
before anyone spends a backtest on them?

**Before this change: no.** Each backtest stood alone. The strategy translator is deliberately blind to
results (so it cannot tune a spec to the data), the interpreter judged one run at a time, and nothing
accumulated across runs.

**Now: yes.** Every backtest, holdout test and paper-trading record is evidence about a *hypothesis*.
The platform keeps all of it, failures included, in a per-market ledger. From that ledger it derives
Bayesian beliefs that related hypotheses share, and a judge reads them before each new test.
`aitrading research` runs the whole cycle: propose, judge, test, learn, and finally an out-of-sample
exam.

```
 hypothesis (words + testable spec)
      │
      ▼
 judge: forecast BEFORE the test  ──► recorded (pre-registration)
      │   (Bayesian engine │ Claude reading the ledger │ your trained model)
      ▼
 backtest on the research window
      │
      ▼
 ledger: evidence (+ every earlier run) ──► beliefs for ALL hypotheses (tested or not)
      │                                         │
      │                                         ├─► feedback / critique to the generator (next round)
      │                                         └─► calibration: was the forecast any good?
      ▼
 best survivors ──► holdout exam on unseen years ──► paper trading (new data, forever)
```

## 1. What is learned, and how

### Hypotheses, not specs

A spec is one way to test a claim. The claim is the hypothesis: the signal and its direction,
filters, timing rule, universe, rebalance frequency, long-only or long-short. Quintiles vs deciles,
cost assumptions, dates, names and the attribution model are implementation details. All runs of one
claim are measurements of the same thing. Each hypothesis has an id (`h-…`), a plain-English statement
("Stocks with high 3-day return outperform … [3-day momentum]") and, when a language model proposed
it, a mechanism.

### Beliefs: one Gaussian process over hypothesis space

Each hypothesis `h` has an unknown gross Sharpe ratio `g_h` (before costs) and a cost drag `c_h`
(annual trading cost / volatility). What you earn is `g_h − c_h`.

* **Related claims share evidence.** Every catalog feature has a family (past returns, value,
  profitability, risk, …), an orientation and, where it matters, a horizon. A kernel built on that
  geometry gives, for example:
  * 3-day vs 5-day momentum: similarity 0.67;
  * 3-day vs 10-day momentum: 0.29;
  * 3-day momentum vs 12-1 momentum: 0;
  * 3-day momentum vs 3-day reversal: −1 (the opposite bet).

  The kernel is positive semi-definite, so beliefs about all hypotheses form one Gaussian process
  `g ~ GP(m0, s0²·k)`.
* **Costs are modelled separately.** Opposite bets share their trading cost. If 3-day momentum loses
  1.1 net of costs, 3-day reversal does *not* earn +1.1: both pay about 1.2 Sharpe units of costs.
  Costs come from each run's turnover (2 × turnover × rebalances × bps / volatility). For untested
  ideas they are estimated from similar runs.
* **Re-testing the same data is not new evidence.** Measurement errors of two runs are correlated by
  how much their monthly returns overlap and co-move (ρ × n_common / √(n₁n₂)). A re-run, or a decile
  version of a quintile test on the same years, barely moves a belief. The same claim on new years (a
  holdout, paper trading) moves it for real.
* **Every run counts, failures included.** The posterior conditions on all trials. That handles the
  "I tried 20 variants and kept the best" problem for related variants: the winner is pulled back
  towards its failed neighbours.
* **Skepticism is learned.** How good a new idea is expected to be (`m0`) and how much ideas differ
  (`s0`) are estimated from the ledger by empirical Bayes. In a market where most ideas failed, `s0`
  shrinks and in-sample successes are discounted more.
* **Year-to-year drift is in the noise.** A strategy's true Sharpe drifts between periods
  (`REGIME_SD` = 0.35 per year), so even ten years of backtest leave real uncertainty (about ±0.27).

The result is a belief for every hypothesis, tested or not: net Sharpe mean ± sd, P(>0), P(>0.3),
status (supported / refuted / uncertain / untested), and the years of in-sample and out-of-sample
evidence behind it.

### The judge: "intuition" as a forecast

Before a test, a judge forecasts its result. The forecast includes the related experience and a
written critique, and it is stored (pre-registered) so it cannot be adjusted afterwards. Three judges
share one interface:

| Judge | What it is | Learns how |
|---|---|---|
| `bayes` (default) | The belief engine's posterior + nearest tested neighbours + critique | Exact Bayesian updating on the ledger; no language model, so it cannot be swayed by wording |
| `claude` | Claude reads the hypothesis, its mechanism and the experience digest. It gives a *literature prior* (from economics and papers, formed before reading the log) and an overall view | In-context: the retrieved ledger is in its prompt. No training. The literature prior enters the engine as one weak pseudo-observation (sd ≥ 0.5), so data dominates. Cited ids must exist in the log; numbers not in the log are flagged |
| `open` | A model you trained (below), served behind an OpenAI-compatible endpoint (vLLM, llama.cpp, Ollama …) | RL on the ledger's outcomes |

`aitrading beliefs calibration` scores every judge's resolved forecasts:

* Brier skill against a coin flip;
* log score against a naive prior;
* coverage of the 80% interval.

These scores answer "is this intuition any good?" with data. Out-of-sample rows (holdout, paper) are
the ones that matter.

## 2. Worked example (the simulated market)

This is the 3-day / 10-day momentum example, run offline on the built-in simulated market (500 names,
10 years, weekly rebalancing, 10 bps one-way costs):

```text
$ aitrading backtest --provider synthetic "3-day momentum"
Strategy: CAGR -16.92% | Sharpe -1.14 | …
Learning: [h-9b38da751a50] Stocks with high 3-day return outperform … [3-day momentum]
  Before the run, bayes predicted Sharpe -0.80 ± 0.78 (P(>0) = 15%; novel; 0 related hypothesis(es) tested).
  Observed -1.14 (surprise z = -0.4). Belief now: net Sharpe -1.16 ± 0.27, P(>0) = 0%, refuted

$ aitrading backtest --provider synthetic "10-day momentum"
  Observed -0.56 … Belief now: net Sharpe -0.69 ± 0.26, refuted
  Related: [h-9b38da751a50] 3-day momentum (similarity +0.29): -1.16 ± 0.27

$ aitrading backtest --provider synthetic "12-1 momentum deciles"
  Observed +0.77 … Belief now: net Sharpe +0.48 ± 0.27, P(>0) = 96%, supported

$ aitrading beliefs judge --market synthetic "5-day momentum"
Prediction: Sharpe -0.81 ± 0.45 net of costs (P(>0) = 3%); gross +0.16 minus an estimated cost drag of 0.97 at 10 bps.
Related experience: 3-day momentum (similarity +0.67): -1.12; 10-day momentum (+0.57): -0.66
Costs decide it: the expected gross Sharpe (+0.16) is smaller than the cost drag (0.97) …
Verdict: CONTRADICTS EXPERIENCE

$ aitrading beliefs judge --market synthetic "3-day reversal"
Prediction: Sharpe -0.92 ± 0.34 net of costs; gross +0.14 minus an estimated cost drag of 1.06 at 10 bps.
Related experience: 3-day momentum (similarity -1.00, the opposite claim): -1.12

$ aitrading beliefs judge --market synthetic "6-month momentum"
Prediction: Sharpe +0.28 ± 0.40 net of costs (P(>0) = 76%) … Related: 12-1 momentum (+0.62): +0.48
Verdict: consistent with experience.

$ aitrading beliefs lessons --market synthetic
- Momentum, horizons of 2 weeks or less has not worked here: 2 variant(s) tested, 0 supported, 2 refuted …
- Momentum, 6-12 month horizons has worked here: 1 variant(s) tested, 1 supported …
```

The untested 5-day momentum inherits the experience of its neighbours. The opposite bet does not
inherit a mirror-image success, because costs break the symmetry. 6-month momentum borrows credibility
from 12-1 momentum. All of it comes with uncertainty that shrinks only with genuinely new data. (These
numbers come from a simulated market and say nothing about real markets.)

## 3. The research loop

```bash
aitrading research "short-term momentum and reversal" --rounds 3 --per-round 3 --holdout-years 2
aitrading research --generator claude --judge claude          # Claude proposes and judges
```

Each round runs four steps:

1. **Propose.** A generator proposes hypotheses. Claude reads the experience digest and last round's
   feedback, and writes claims with mechanisms; the strategy translator makes them testable. The
   offline generator instead offers the idea library plus every return horizon × momentum/reversal.
2. **Judge.** The judge forecasts each candidate.
3. **Test.** The loop tests the candidates with the highest mean + sd: promising or still very
   uncertain ideas first, ideas that contradict experience last. Within a round, an idea similar to
   one already chosen loses its uncertainty bonus, so one round does not test an idea and its
   near-copy (or its mirror). This is Bayesian optimisation over hypothesis space.
4. **Learn.** Tests run on the research window only. Each result updates the beliefs, and the line
   "predicted X, observed Y, belief now Z" plus the critique becomes the next round's feedback.

After the rounds, the best survivors sit a holdout exam on the last `--holdout-years`, which no
round looked at. The ledger counts holdout uses. A second loop on the same window is told the holdout
is burnt, and its "holdout" results are recorded as in-sample. Re-testing a hypothesis on the same
research window is skipped. The report (`aitrading_output/research/<time>/report.md|json`) lists the
predictions, observations, surprises, holdout results, final beliefs, lessons and the trial count.

## 4. Training your own judge (RL on an open model)

```bash
aitrading beliefs export judge.jsonl --test-from 2023-01-01 --llm-cutoff 2024-06-01
python scripts/train_judge_grpo.py judge.jsonl --baseline-only     # the score to beat
python scripts/train_judge_grpo.py judge.jsonl --model Qwen/Qwen2.5-1.5B-Instruct --lora
vllm serve judge-ckpt
aitrading research --judge open --judge-url http://localhost:8000/v1 --judge-model judge-ckpt
```

* **Examples are point in time.** Each piece of evidence becomes one example. Its prompt holds the
  experience digest *as it was before the result was recorded*. For holdout and paper targets, only
  evidence whose data ended before the target period began is included. The prompt format is exactly
  what `OpenAICompatibleJudge` sends in use.
* **The reward is a proper scoring rule.** It is the log score of the stated P(Sharpe > 0) for what
  happened, relative to a coin flip: `log2(p) + 1` if positive, `log2(1−p) + 1` otherwise. The model
  is paid for calibration, not optimism. Unparseable output scores −1.
* **Each example carries a baseline.** It records the Bayesian judge's probability in the same
  context. A trained judge is worth serving only if it beats that on the later (test) examples.

### When it is worth it, and when it is not

| Approach | Data needed | Cost | Risk |
|---|---|---|---|
| Bayesian engine (default) | Works from the first run | Free, exact | Knows only what the kernel encodes (catalog geometry) |
| Prompted LLM judge | Works from the first run | API calls | Uncalibrated confidence; has read papers about the backtest period (look-ahead). Use it for mechanisms and semantics, and let `calibration` decide how much to trust it |
| RL-trained open judge | Hundreds to thousands of resolved hypotheses, ideally out of sample | GPU + engineering | Noisy, correlated rewards are easy to overfit. Train on early periods, test on later ones |

The reward is noisy: one year of returns pins a Sharpe ratio down only to about ±1. Outcomes within a
period are correlated: a momentum regime lifts every momentum variant at once. Most ledgers will
therefore not have enough independent outcomes to train a judge for a long time. Start with the
engine, add the Claude judge, and watch `calibration`. Train when the data say the prompted judge is
miscalibrated and the export has enough out-of-sample targets.

## 5. Pitfalls the design guards against, and the ones it cannot

* **Data snooping.** All trials are recorded and related trials share evidence. Forecasts are
  pre-registered, a holdout is spent once, and the trial count is reported. Unrelated lucky winners
  are only shrunk towards the learned base rate. The holdout and paper trading are the real guard.
* **Overlapping data.** Correlated measurement errors stop repeated tests on the same years from
  inflating confidence.
* **Costs.** Beliefs and forecasts are net of costs at the spec's cost level. The gross effect and
  the cost drag are shown separately.
* **Pretraining look-ahead.** A language model "knows" which anomalies worked before its cutoff.
  Its literature prior is a weak, separately scored input, and the export flags examples after a
  given cutoff.
* **Non-stationarity.** Year-to-year drift is part of the noise model, so beliefs never become
  certain from history alone; paper trading keeps adding new evidence (`strategy run` records it
  automatically).
* **Survivorship and data quality.** Evidence keeps the run's survivorship / point-in-time caveats.
  The engine cannot remove a bias that is in the data itself.
* **Markets are separate.** Each data provider has its own ledger, so the simulated market never
  moves beliefs about real ones.

## 6. Commands

```bash
aitrading backtest "3-day momentum"            # records the run (add --no-learn to skip), prints what it taught
aitrading beliefs list [--status refuted]      # every tested hypothesis and its belief
aitrading beliefs show h-9b38                  # evidence, forecasts, literature prior
aitrading beliefs judge "5-day momentum"       # forecast + related experience, before testing
aitrading beliefs lessons                      # per-concept lessons + the digest the agents read
aitrading beliefs calibration                  # how good each judge's forecasts were
aitrading beliefs export judge.jsonl           # training data for your own judge
aitrading research ["theme"]                   # the loop: propose, judge, test, learn, holdout exam
aitrading strategy run NAME                    # paper trading; >= 3 months of record becomes out-of-sample evidence
```

Ledgers live in `~/.aitrading/beliefs/<market>/ledger.json` (`$AITRADING_HOME/beliefs/…`). Code:
`src/aitrading/beliefs/`, which contains `semantics`, `ledger`, `bayes`, `judge`, `digest`,
`learner`, `loop`, `calibration` and `dataset`.
