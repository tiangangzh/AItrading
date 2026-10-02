# ADR-001: Where the LLM reasoning layer sits relative to Bloomberg, LSEG and S&P Capital IQ

| | |
|---|---|
| **Status** | Accepted. Phase 1 can go ahead now. Phases 2 and 3 wait on the written vendor and legal gates G1 to G6 below. |
| **Date** | 2026-10-02 |
| **Supersedes** | none |
| **Companion** | [`VENDOR_REFERENCE.md`](VENDOR_REFERENCE.md): entry points, push-down grammar and field maps with verification status |
| **Model** | Anthropic `claude-opus-5-5`, pinned. Effort is set explicitly to `high`, because the default on Opus 5.5 is `medium`. |

Confidence labels: **[H]** means verified in vendor-authored code or docs. **[M]** means verified only through search extracts or community code. **[L]** means not verified (inference, or a contract not reviewed). **UNVERIFIED** marks anything the fact-check pass could not confirm. Adapters must treat it as configuration, never as a constant.

---

## 1. Context

We are building a Python pipeline. It takes an analyst's observation, for example: *"US $2-20bn names whose 50d SMA is still above the 200d, down 20-45% from the 52-week high on heavy volume, RSI(14) < 45, FCF yield >= 5%, trailing revenue growth >= 10%, short interest >= 5% of float."* It screens the US equity universe, ranks the survivors, reads each survivor's latest earnings call, and explains the dislocation. The open question is whether the reasoning layer sits **inside** a terminal (Bloomberg ASKB/BQuant) or **on top** of the vendors, with a frontier model fed through LSEG or S&P.

The verified research gives five facts that drive the decision:

1. **Only the vendors' native interfaces can screen the universe.**
   - The native screening interfaces are BQL `filter(equitiesuniv(['ACTIVE','PRIMARY']), ...)` [H], LSEG `SCREEN(...)` / `lseg.data.discovery.Screener` [H], and SQL over S&P Xpressfeed/Snowflake [M].
   - No vendor MCP server screens:
     - Kensho/kFinance has 38 tools, all scoped to an identifier [H].
     - The LSEG LFA plugin has 22 tools and no screen [H].
     - Bloomberg Enterprise MCP on DL+ (launched 2026-09-29) caps the securities and fields per call [M].
   - The S&P GDS REST API has no screen function, and its observed limit is 10,000 requests a day [M].
2. **ASKB is strong but closed.** It cites its sources, emits the BQL behind every data answer, and supports saved Workflows [M]. The fact-check found no public ASKB API. That is an absence of evidence, not a confirmed fact. Bloomberg routes each query to a model it chooses (Fortune, 2026-04-28) [M], so the firm cannot pin a model, run evals against it, or keep its own prompt and output logs.
3. **Data rights are the binding constraint, and none of the key contracts has been reviewed.**
   - **Bloomberg.** Terminal, Desktop API and Excel data is licensed per user. Library guides say it "may not leave the local machine" [L: the contract was not reviewed].
   - **S&P.** S&P markets the Kensho LLM-ready API "for use with customer's GenAI application of choice" (2024-11-13 press release) [M: the contract text was not reviewed].
   - **LSEG.** The LSEG MCP server enforces entitlements [H]. The often-quoted "licenses the what, not the how" comes from the London Stock Exchange advisory on exchange data. It does not govern LSEG D&A content such as I/B/E/S, StarMine, Reuters or StreetEvents [M].
   - **DL+.** Enterprise MCP lets the client bring its own model. That is **not** the same as permission to send returned values to Anthropic [L].
4. **Anthropic platform constraints [H]:**
   - Opus 5.5 is eligible for zero data retention (ZDR). The Messages-API MCP connector is **not**.
   - Citations combined with `output_config.format` return a 400.
   - A forced `tool_choice` returns a 400 on Opus 5.5.
   - Server-side refusal fallback (`server-side-fallback-2026-07-01`, `fallbacks='default'`) is not supported on the Batches API.
5. **Regulation.**
   - SR 26-2 (2026-04-17) supersedes SR 11-7 and explicitly excludes generative and agentic AI [M].
   - FINRA's 2026 Oversight Report recommends prompt/output logs and human checkpoints [M].
   - Advisers Act s.204A requires MNPI policies.
   - The SEC's 2023 predictive-analytics proposal was withdrawn on 2025-06-17 [M].

## 2. Decision drivers

| # | Driver | What "good" looks like |
|---|---|---|
| D1 | Data rights and compliance | No vendor value reaches a third-party model without written permission. Every datum has a provenance tag. MNPI sources are gated. |
| D2 | Numerical correctness and point-in-time | Every number is computed deterministically and traced to a field code. No silent NaN wipe-outs. Explicit as-of dates. Short-interest and growth filters are applied before top-N truncation. |
| D3 | Auditability and model risk | The model is pinned. Prompts, outputs and request IDs are logged. The screen and rank are deterministic, so they can be validated and backtested without LLM look-ahead. |
| D4 | Operability and cost | The pipeline can be scheduled. It can be tested in CI from recorded fixtures. Its build surface is bounded and it does not exhaust vendor capacity. |
| D5 | Portability | We can swap the LLM or add or drop a data vendor without redesigning the pipeline. |
| D6 | Analyst fit | It works with how PMs already use the Terminal, including ASKB and broker research. |

## 3. Options considered

**A. Reasoning inside Bloomberg** (`inside-bloomberg`)
- BQL in BQuant computes every number from a vetted template filled from a typed ScreenSpec.
- ASKB Workflows, Document Insights and Textual Analytics read the transcripts, news and broker research inside the licence boundary.
- Claude appears only on data-free paths (natural language to ScreenSpec), or later through DL+ Enterprise MCP once that is cleared.
- Honest summary: it has the cleanest licensing position and the only access to broker research and Bloomberg News. It does not deliver an end-to-end pipeline, because the narrative step is manual, the model cannot be pinned, the firm holds no logs, and nothing can be tested in CI.
- Fatal flaws the judges found:
  - Short interest is filtered only *after* `grouprank() <= 25`, so qualifying names ranked 26th or lower are lost.
  - Step 2 filters on the unverified `free_cash_flow_yield()`, which contradicts the design's own rule that unverified items are used only for ranking.
  - Survivor tickers and ranks are stored firm-side, yet the design claims it needs no approval.
  - ASKB text cannot be retained for books-and-records without risking a dissemination breach.

**B. Reasoning on top of LSEG and S&P, vendor-neutral, Bloomberg optional** (`on_top_lseg_spglobal_vendor_neutral`)
- A Python orchestrator runs on an LSEG Platform session (SCREEN plus history) with Kensho for transcripts and line items.
- Technicals are computed in pandas, LSEG values are reconciled against S&P, and Claude writes narratives grounded with `search_result` citations.
- Honest summary: it has the strongest numeric controls (preflight, attrition table, per-run snapshots, reconciliation) and keeps model control with the firm. It throws away Bloomberg's one-request point-in-time screen, which the firm already pays for, and it needs new LSEG and Kensho contracts.
- Fatal flaws the judges found:
  - Step 7 sends Reuters headlines and LSEG field values to Anthropic, which contradicts its own licensing section.
  - No verified short-interest or put/call field exists at LSEG or S&P.
  - `get_history` corporate-action adjustment is unverified.

**C. Reasoning on top, compute pushed down into each vendor, licensing-aware boundary, ASKB alongside** (`ontop_reasoning__pushdown_compute__licensed_boundary`)
- Claude emits a typed ScreenSpec and writes cited prose. It never computes a number or writes BQL, SCREEN or SQL.
- A deterministic compiler pushes only verified predicates down to each vendor. One canonical evaluator re-checks every predicate.
- A deny-by-default licence-class table (L0 to L7) decides what may reach the model. ASKB stays an analyst tool and is not a pipeline stage.
- Honest summary: it is the best long-run position on rights, audit and lock-in.
- Fatal flaws the judges found:
  - `composite_v1` uses Bloomberg values in zone E, which its own boundary forbids.
  - There is no cross-vendor symbology map.
  - "Same evaluator inside BQuant" is assumed, not evidenced.
  - "Works end to end" is overstated: on a Terminal-only desk, even tickers need written approval before they leave zone T.
  - Three compiler dialects at once is too much build.

| Lens (judge) | A. Inside Bloomberg | B. On top, LSEG/S&P | **C. On top + push-down + boundary** |
|---|---|---|---|
| Licensing, compliance, MNPI, model risk, audit, prompt injection | 6.5 | 5.5 | **8.0** |
| Numerical correctness, point-in-time, coverage, expressiveness | 6.0 | **6.5** | 6.0 |
| Operability, cost, lock-in, CI, swappability, team fit | 5.0 | 6.0 | **6.5** |
| **Total (of 30)** | 17.5 | 18.0 | **20.5** |

## 4. Decision

**We adopt Option C, built in phases, with the grafts listed below.**
- Numbers are computed where the data is licensed to live. Bloomberg values are computed in BQuant from one vetted BQL template.
- Words are generated by a pinned, logged Claude, only from text that is GenAI-licensed or public.
- A deny-by-default policy table under change control draws the line between the two. Every row cites the clause or written confirmation it relies on.
- ASKB stays in analysts' hands as the reader of broker research and Bloomberg News. Only the analyst's own judgement of an ASKB answer is recorded, never ASKB text.

**Phasing.** We build one compiler dialect at a time.
- **Phase 1 (now; needs no new contract).** This is the inside-Bloomberg pattern with Option C's controls.
  - Claude turns the observation into a ScreenSpec. It sees only the analyst's sentence and our feature catalogue.
  - A BQL template runs in BQuant. Residual predicates, short interest (BDP on the same workstation), the rank and the attrition table are all computed **inside zone T**.
  - Each name is explained through a firm-shared ASKB Workflow, followed by mechanical re-checks.
  - Nothing derived from Bloomberg data leaves zone T, including tickers and ranks, until gate G1 clears.
- **Phase 2 (after G1 and G2).** Only survivor identifiers, ranks and pass/fail booleans cross from T to E.
  - Zone E pulls Kensho transcripts, Kensho line items and EDGAR text for those names.
  - Claude writes cited narratives. A verifier and a human review them.
- **Phase 3 (each step gated separately).**
  - Add the LSEG or S&P compiler only when that contract exists.
  - Use DL+ Enterprise MCP for per-name drill-down after G4.
  - Loosen L3 classes one vendor at a time as written confirmations arrive.

**Grafts from the runners-up, and fixes to C's own flaws:**
1. *(A)* One vetted, git-versioned BQL template, with no general BQL compiler in Phase 1.
   - A **field-admission gate**: an item may be used as a filter predicate only after a BQLX, FLDS or preflight test on a known security. The test is recorded in `field_validation_log` (item, test ticker, date, value, units).
   - Items with unverified units enter **only as z-scores** until they are admitted. The compiler enforces this.
2. *(A, B)* A strict order of operations. Pushed predicates run first. Then every residual predicate runs, **including short interest, on the full survivor set**. Only then are survivors ranked and truncated to top N. No filter is ever applied after `grouprank`.
3. *(B)* NaN never passes a predicate.
   - Each predicate gets its own attrition count. BQL's nested `and_` cannot provide that, so we evaluate the survivor panel in pandas.
   - A field that is missing across the board suspends its leg, prints a `LEG NOT EVALUATED` banner, and the run continues. It never returns zero names, and it never uses `fillna(0)`.
4. *(A, B)* Every date range is anchored to an explicit absolute `as_of`, never `'0D'`, and the as-of is logged.
   - Backtests use `members('<index>', dates=...)`, because `equitiesuniv(dates=)` is UNVERIFIED.
   - US short interest always carries `SHORT_INT_DT`.
5. **Fix to C:** the rank is computed in the zone that holds its inputs.
   - `rank_v1` uses Bloomberg inputs and runs in zone T. Only its output crosses.
   - Zone-E values (for example Kensho FCF) may produce **reconciliation flags** but never silently change the rank.
6. **Fix to C:** a maintained symbology table keyed on Bloomberg ticker, RIC/PermID and CIQ/Kensho company id, validated by name, market cap and earnings date, is required before any cross-vendor join. If an id fails to resolve, the name is dropped and flagged.
7. **Fix to C:** the canonical evaluator is one pure pandas/numpy module, copied into the BQuant project with a recorded SHA-256 that CI checks. Whether BQuant Desktop can import a firm package is UNVERIFIED. If it cannot, the vendored copy is the supported route.
8. *(A)* "Delete, don't soften". Any number or quote that cannot be verified is removed from the memo.
   - Every number ASKB cites is re-run from the BQL it emitted, at the same `as_of`.
   - BQL that ASKB emits enters the template only after a human checks it in BQLX.
9. *(Judges)* The quarantined transcript reader is **mandatory**, following Anthropic's earnings-reviewer pattern. It has no tools and no MCP, and its JSON is regex-constrained with `additionalProperties:false`. The citation-grounded writer runs after it.
10. *(B)* Each run produces a per-run parquet snapshot plus hashes wherever the licence allows. For Terminal data the run stores only the BQL string, the template SHA and the absolute dates. CI replays recorded fixtures offline. An NL-to-spec eval set and the citation pass rate are re-run whenever the model id changes.
11. *(Judges)* Integration tests cover every Claude call shape:
    - citations combined with a format (expect 400);
    - forced `tool_choice` (expect 400);
    - fallback header pairing;
    - whether `messages.parse` merges `output_config.effort` with `output_format` (UNVERIFIED).

We do **not** claim the pipeline is automated end to end today. In Phase 1 the narrative step is analyst-driven by design.

## 5. Consequences

**Positive**
- It runs today on assets the firm already licenses (Terminal, BQuant, ASKB), with no new contracts.
- The licensing posture is the strictest of the three options, and loosening it is a one-row configuration change plus a citation to the confirming document.
- The screen and rank are deterministic, point-in-time and backtestable, and they sit in the SR 26-2 model inventory. The LLM steps are governed separately under the firm's GenAI policy.
- The ScreenSpec belongs to us, so LSEG or S&P can be added later as another compile target without rewriting prompts.

**Negative**
- Phase 1 has no automated narrative. Analyst time grows linearly with the length of the shortlist.
- Code and data in zone T cannot be tested in firm CI except through vendored-module hash checks and recorded fixtures of *our* code paths.
- Phase 2 may mean buying data twice (screen on Bloomberg, re-source the LLM-visible metrics from Kensho).
- The feature catalogue and admission log are a permanent maintenance cost.

**Risks and mitigations**

| Risk | Mitigation |
|---|---|
| An unverified item (`free_cash_flow_yield()`, bare `smavg(period=N)`, `cntry_of_domicile()`) returns NaN and empties the screen | Admission gate. Per-predicate counts. The run fails if any stage unexpectedly returns 0. |
| Short-interest fields (all community-sourced) fail or need a securities-finance entitlement | Pull via BDP after an FLDS check. Abort on `fieldExceptions`. Show the leg as `NOT EVALUATED`. Compute SI % of float as `SHORT_INT/EQY_FLOAT` locally. |
| Units: BQL `total_return` is a fraction; LSEG `TR.TotalReturn3Mo` is in percent; `free_cash_flow_yield` and `pct_chg` units are UNVERIFIED | The FieldMap records unit and sign. The compiler scales thresholds. Unverified units are allowed only in z-scores. |
| Semantic substitution: `sales_growth(fpt='A',fpo='1')` is FY1 *forward*; `cntry_of_risk` is economic exposure, not listing | Separate catalogue features (trailing vs forward growth; risk country vs listing). The compiler refuses a mapping that changes the meaning. |
| BQL capacity limits and timeouts on universe queries | Static filters first in nested `filter()`. `with_params={'mode':'cached'}`. Tranches. Abort on the first capacity error. |
| A boundary misclassification leaks data | Deny by default. Policy-table unit tests. No live MCP in the batch path. Legal sign-off on every row. |
| Narrative hallucination or prompt injection | Mandatory quarantined reader. No tools in the writer. Search results marked untrusted. `verify_response` checks each number against the metric it claims to be. Human review before use. |
| Wrong earnings call (fiscal vs calendar quarter, or duplicate versions) | Match on the call name. The call date must fall inside the drawdown window. Keep only the latest `transcriptCreationDateUTC` per `keyDevId`. |

## 6. Reference data flow for the representative task

```
 ZONE E (firm)                 ZONE M (Anthropic, ZDR org,         ZONE T (Bloomberg licence: Terminal/BQuant)
                               inference_geo='us')
 [1] observation ----------->  ScreenSpec (structured output)
     + feature catalogue  <--  (sees: words + catalogue only)
 [2] validate spec, analyst
     approves pushed/residual
     split; git: template SHA
              |  spec JSON + template SHA (code/metadata only)
              +------------------------------------------------------> [3] BQL push-down, absolute as_of
                                                                       [4] residual eval (vendored pandas)
                                                                           + BDP short interest (all survivors)
                                                                       [5] rank_v1 in zone T -> top N
                                                                       [6] ASKB Workflow per name (Phase 1)
                                                                           + re-run emitted BQL; check quotes
   (Phase 2, after G1) <------ survivor ids + ranks + booleans ONLY ---+
 [7] Kensho transcript / EDGAR
     -> MNPI + licence gate
 [8] quarantined reader ----->  JSON (no tools)
 [9] evidence pack ---------->  cited narrative (no tools, no format)
 [10] verify_response + human   (sees: L0/L1 text + L0/L1 metrics)
 [11] append-only audit JSONL
```

1. **Observation to ScreenSpec. Claude involved; no vendor data.**
   - Call `client.messages.parse(model='claude-opus-5-5', max_tokens=16000, output_format=ScreenSpec, output_config={'effort':'high'}, system=<catalogue>, messages=[observation])`.
   - The features form a closed enum and include `revenue_growth_ltm_yoy` as distinct from `revenue_growth_fy1_fwd`. Anything the catalogue cannot express goes into `unmapped_requests`.
   - Code then checks the bounds, that each condition has exactly one right-hand side, and that `stop_reason` is not `refusal` or `max_tokens`. **The LLM does not see or produce any number from the market.**
2. **Compile. Deterministic.**
   - The FieldMap lists, for each feature: its vendor expression, unit, sign and admission status.
   - Only admitted predicates are pushed. The analyst approves the pushed/residual/unmapped lists, and the approval is logged.
3. **BQL push-down in BQuant (zone T).**
   - Universe: `filter(filter(equitiesuniv(['ACTIVE','PRIMARY']), cntry_of_risk()=='US' and cur_mkt_cap(currency='USD') >= 2e9 and cur_mkt_cap(currency='USD') <= 2e10), ...)`.
   - Moving averages: `smavg(px_last(dates=range(as_of-2Y, as_of), fill='prev', ca_adj='full'), period=50|200).last('1')`.
   - Momentum: `rsi(close=px_last(), period=14)`.
   - 52-week high: `maxmin(px_high, px_low).with_additional_parameters(period=260)['maxline']`.
   - Volume: `px_volume(...).dropna().last(5).avg()` against the 60-day average.
   - Executed via `bql.Request(..., with_params={'fill':'prev','mode':'cached'})`. *The vendor computes every number in this step.*
4. **Residual predicates (zone T, vendored evaluator).**
   - FCF yield and trailing growth are fetched as items on the survivors only. They become thresholds only after admission. Until then they enter only as z-scores, and the gap is flagged.
   - Trailing growth: `sales_rev_turn(fpt='LTM')` divided by the same item at `dates=<as_of-1Y>`. Admission is pending.
   - Short interest for **all** survivors: Desktop API `ReferenceDataRequest` on `//blp/refdata` for `SHORT_INT`, `SHORT_INT_RATIO`, `SHORT_INT_DT`, `EQY_FLOAT` (all UNVERIFIED, FLDS first).
   - Run per-predicate attrition. NaN never passes. *The deterministic code computes everything here; no LLM.*
5. **Rank (zone T).**
   - `rank_v1 = z(fcf) + z(revenue growth) + z(contributor_revisions NETUP 12w) − z(RSI) − z(min(ND/EBITDA, 6)) + 0.5·z(SI%)`. Weights are versioned.
   - BQL `groupzscore`/`grouprank` run on the same zone-T inputs as a cross-check.
   - Enrichment columns: `implied_volatility(expiry='30d', delta='50')`, `volatility(calc_interval='252d')`, `best_target_price()`.
6. **Phase 1 narrative (zone T, analyst).**
   - The analyst runs the saved ASKB Workflow "Post-earnings dislocation review".
   - Each cited number is re-run from the BQL ASKB emitted. Each quote is opened in CF/DS and confirmed verbatim. Anything that fails is deleted.
   - The firm-side record holds only the Workflow ID, analyst, date and the per-claim pass/fail. Recording ticker-level outcomes firm-side is itself gated on G1.
7. **Phase 2 evidence (zone E).**
   - Kensho: `Client(client_id, private_key).ticker(sym).company.latest_earnings` gives the call name and `key_dev_id`, then `.transcript.raw`. Requires `TranscriptsPermission`.
   - The call must fall inside the drawdown window.
   - Text is split into speaker paragraphs, each hashed with SHA-256 and tagged L1 (pending G2).
   - Latest 10-Q and 8-K from EDGAR, tagged L0.
   - Expert-network and data-room sources are blocked.
8. **Quarantined reader. Claude with no tools and no MCP.**
   - Returns regex-constrained JSON: guidance changes and paragraph IDs.
   - Text in = untrusted transcript. Numbers out = only strings matched against the source.
9. **Narrative. Claude.**
   - Call `client.messages.create(..., output_config={'effort':'high'}, inference_geo='us')`.
   - Inputs: transcript paragraphs as `search_result` blocks with `citations: {enabled: true}`, plus a metrics table containing **only L0/L1 values** (Kensho market cap and line items, and indicators computed locally from them). Bloomberg-derived facts appear only as booleans or ranks.
   - No tools and no output format.
   - Optional second pass: `messages.parse(output_format=Dislocation)` over the verified prose.
10. **Verify.**
    - `cited_text` must be found in the normalised corpus.
    - Every number in the prose must match the metric it claims to be, in the table or in cited text.
    - On failure, regenerate once, then send to the human queue marked UNVERIFIED.
    - The human reviews it, and any expert-network citation is routed to compliance.
11. **Audit.** Write one append-only JSONL record per run (section 7).

**Where the LLM is allowed to touch numbers: never as a producer.** It may *restate* a number only if that number is in the supplied L0/L1 metrics table or in cited text, and the verifier enforces this. Every screened, ranked or compared number comes from a vendor push-down or from the deterministic evaluator.

## 7. Licensing and compliance boundary

| Class | Examples | Computed where | May reach Claude | Evidence required before use |
|---|---|---|---|---|
| L0 Public | SEC EDGAR | anywhere | full text | none |
| L1 GenAI-licensed | Kensho LLM-ready API (transcripts, line items, capitalisation) | zone E | values and text, inference only, **pending G2** | written S&P confirmation covering Anthropic as processor, log retention and internal memos |
| L2 Vendor MCP | LSEG MCP, Kensho MCP, Bloomberg Enterprise MCP (DL+) | vendor | analyst-interactive only, never in the batch path. The MCP connector is not ZDR-eligible. | per-vendor written terms (G4 for DL+) |
| L3 Enterprise, AI use unconfirmed | LSEG Platform (I/B/E/S, StarMine, Reuters, StreetEvents), CIQ GDS, Xpressfeed/Snowflake, DL/DL+ bulk, BQuant Enterprise | firm servers or cloud per contract | ranks and booleans only (derived data may itself be licensable) | written vendor confirmation per dataset |
| L4 Desktop-licensed | Terminal, Desktop API, Excel BQL, BQuant Desktop; LSEG Workspace | that user's workstation | **nothing**; ids, ranks and booleans only after G1 | written Bloomberg approval |
| L5 Black-box only | Bloomberg EDF Textual News | never | never | n/a |
| L6 Contributor-restricted | broker research (BRC, ASKB's 800+ providers), LSEG aftermarket | Terminal/ASKB only | never | n/a |
| L7 MNPI risk | Third Bridge, Guidepoint, Intralinks, Egnyte | blocked | never, unless compliance clears the source | compliance sign-off per source |

**Rules**
- Every datum carries `(vendor, channel, licence_class, field, as_of, mnpi_class)`.
- The policy table lives in git under change control and has CI unit tests. Each row names the contract clause or written confirmation it relies on, the reviewing lawyer and the date. A row with no citation is evaluated as **deny**.
- Logs that contain vendor content inherit that vendor's retention and purge rules.
- No training or fine-tuning on any vendor data, and no shared transcript vector store beyond licensed users.
- Memos stay internal unless redistribution rights exist.

**Prohibited**
- `//blp/bqlsvc` with `clientContext.appName='EXCEL'`.
- Lifting Workspace `edp-token`s.
- Disabling SFTP host-key checks.
- Running a Workspace desktop session on a server.
- Pasting Terminal values or ASKB text into any Claude prompt, email or deck.

**Audit record (per run, append-only JSONL, firm books-and-records retention)**
- Run identity: run ID, UTC timestamp, absolute `as_of`, analyst, orchestrator git SHA.
- Model: id, effort, betas and fallbacks, every Anthropic request ID with its `stop_reason` and fallback events.
- Spec: the ScreenSpec and its SHA-256, `unmapped_requests`, the catalogue/compiler/template versions, and the analyst's approval of the compiled query.
- Per vendor:
  - the compiled query string and its hash;
  - the pushed/residual/unmapped split;
  - per-predicate survivor counts;
  - any suspended legs;
  - response hashes;
  - a parquet snapshot where the licence allows, otherwise query plus dates only.
- Data treatment: the boundary decision for every datum in the evidence pack, and reconciliation flags.
- Evidence: transcript IDs (`key_dev_id`, `keyDevId`, `EventStory_Id`), EDGAR accession numbers and paragraph hashes.
- Output and review: the narrative, its citations, verifier results, the regeneration count, the human disposition, and the analyst's ASKB judgement ("consistent" / "contradicts because ..."), never ASKB text.

**Model-risk split**
- Screen, rank and backtests go in the SR 26-2 model inventory: validation, vendor reconciliation, outcome analysis.
- The NL-to-spec and narrative steps go under the firm's GenAI policy: an eval set (spec exact-match rate, citation pass rate, ungrounded-number rate), a pinned model id, and change control.
- LLM output never enters a backtested signal, because of look-ahead propensity (arXiv:2512.23847).

**Gates**

| Gate | Unlocks | Required |
|---|---|---|
| G1 | Phase 2 | Written Bloomberg approval for survivor ids, ranks and booleans to leave zone T and to be stored firm-side |
| G2 | L1 = yes | Written S&P confirmation that Kensho data and transcripts may be processed by Anthropic for inference, with log retention and internal memo distribution |
| G3 | L3 loosening | Per-vendor written AI-use confirmation for each dataset class |
| G4 | DL+ Enterprise MCP | DL+ terms that allow third-party model processing, retention in prompts and logs, no training, and internal redistribution |
| G5 | Each pushed predicate | An entry in the field-admission log (test ticker, value, units) |
| G6 | Production | Anthropic ZDR org confirmed, `inference_geo='us'`, MNPI source allowlist signed off |

## 8. Known unknowns (do not build on these as facts)

- **Bloomberg BQL:**
  - ASKB API: none found (absence of evidence).
  - `free_cash_flow_yield()`, `net_debt_to_ebitda()`, `cntry_of_domicile()`, bare `smavg(period=N)`.
  - Any short-interest item.
  - `value()` broadcast inside `filter()`.
  - `macd()` columns other than `MACD2`.
  - `equitiesuniv(dates=)`.
  - The `mode=cached` string form.
- **Bloomberg Enterprise MCP:** tool names, auth, caps, and whether third-party model processing is allowed.
- **LSEG:**
  - `TR.FreeCashFlow`, `TR.ShortInterest`, `TR.SharesFreeFloat` (third-party evidence only).
  - Sign and scale of `TR.PricePctChg52WkHigh`.
  - `MAVG()`/`RSI()` inside SCREEN.
  - Platform rate limits.
  - Machine OAuth tokens for LSEG MCP.
- **S&P:**
  - Every short-interest mnemonic.
  - The `documents/search` path.
  - `/tokenRefresh`.
  - `IQ_MARKETCAP` units and `IQ_CAPEX` sign.
  - `^SPX` constituents via `GDSHV`.
  - Whether the Kensho JWT works as an Anthropic MCP `authorization_token`.
- **Platform:** whether BQuant Enterprise can make outbound calls to Anthropic, and whether a firm package can be imported in BQuant Desktop.

## 9. When to revisit

Reopen this ADR if any of the following happens:
1. Bloomberg ships a programmable ASKB or ASKB Workflows API with exportable logs, or the announced governed Terminal-subscriber access, with rights for third-party models.
2. Enterprise MCP on DL+ proves to support server-side screening or compute rather than capped per-security retrieval.
3. Contracts clear L3 data for Anthropic. If so, drop the Kensho re-sourcing step; the architecture otherwise stays.
4. The narrative must become intraday or client-distributed. That needs B-PIPE, non-display licences and redistribution rights.
5. The firm drops Bloomberg or adds a second vendor contract. If so, promote the LSEG SCREEN or S&P SQL compiler from Phase 3.
6. SR 26-2's planned AI request for information, or FINRA/SEC guidance, adds explicit validation requirements for LLM components.
7. Any model-id change. Re-run the evals; this does not change the architecture.

**Sources**
- Anthropic:
  - https://platform.claude.com/docs/en/agents-and-tools/mcp-connector
  - https://platform.claude.com/docs/en/build-with-claude/citations
  - https://platform.claude.com/docs/en/build-with-claude/structured-outputs
  - https://www.anthropic.com/news/finance-agents
- Bloomberg Enterprise MCP (syndicated): https://www.morningstar.com/news/pr-newswire/20260929ny57897/bloomberg-launches-enterprise-mcp-to-seamlessly-connect-bloomberg-data-with-clients-enterprise-ai-applications
- Bloomberg ASKB: https://www.bloomberg.com/professional/insights/press-announcement/bloomberg-unveils-askb-roadmap-for-clients-to-augment-their-investment-process-with-agentic-ai/
- Bloomberg-authored BQuant notebooks (mirror): https://github.com/dmitchell28/Prop-Dev-Environments
- Kensho kFinance: https://github.com/kensho-technologies/kfinance
- LSEG:
  - https://github.com/LSEG-API-Samples/Example.DataLibrary.Python
  - https://www.lseg.com/en/media-centre/press-releases/2025/lseg-announces-collaboration-with-anthropic
- Terminal licence reading [L]: https://libhowto.iese.edu/faq/275023
- BLPAPI Core Developer Guide: https://data.bloomberglp.com/professional/sites/10/2017/03/BLPAPI-Core-Developer-Guide.pdf
- EDF Textual News fact sheet: https://assets.bbhub.io/professional/sites/41/Fact-Sheet-EDF-Textual-News.pdf
- Earnings-reviewer cookbook: https://github.com/anthropics/financial-services/blob/main/managed-agent-cookbooks/earnings-reviewer/subagents/transcript-reader.yaml
- Lookahead propensity: https://arxiv.org/abs/2512.23847
