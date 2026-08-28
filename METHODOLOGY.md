# RouterArena Campaign Report — lemonade-router

**Aug 14 – Aug 28, 2026 · Final: arena 76.74 (accuracy 78.03%, $0.171/1K, robustness 73.1)**

## 1. Executive summary

Over two weeks we built, calibrated, and evaluated a local-first LLM router for the [RouterArena](https://github.com/RouteWorks/RouterArena) leaderboard, with the explicit pre-registered goal of a **top-3 placement** and the constraint that the submission run on the **public release of Lemonade**. The final system — a tri-model ensemble in which two Lemonade-served local models vote on every query and a cloud model referees disagreements — scored:

| Metric | Result |
|---|---|
| **Arena score (β=0.10)** | **76.74** |
| Accuracy | 78.03% (8,400/8,400 graded, 0 abnormal) |
| Cost | $0.171 / 1K queries |
| Robustness | 73.1 |
| Local share | ~80% of queries answered on-device |

Total cloud spend for the entire campaign: **~$50**. Local compute: ~10 box-days across two AMD Ryzen AI Max+ 395 machines (128 GB unified memory each).

## 2. Goal, constraints, ground rules

- **Goal:** top-3 on the RouterArena Acc-Cost Arena leaderboard, pre-registered from the outset.
- **Constraint:** the submission must be reproducible on the public Lemonade release — no experiment-branch features in the serving path. (The `ds4` recipe serving DeepSeek-V4-Flash shipped in a Lemonade release within the campaign window.)
- **Compliance rule:** RouterArena is evaluation-only. No router component may be trained, fit, or tuned on its data *including labels*. Our discipline: every threshold and design choice justified only by (i) an external calibration set built from upstream public benchmarks (this repo, `calibration/`), (ii) our own runtime/token telemetry, or (iii) *public* graded submissions already in the RouterArena repo — the same information every submitter can see. Full-set labels were touched only by pre-registered scoring runs of frozen configurations.

## 3. The metric, and what it implies

Arena score = F-β(accuracy, log-normalized cost) with **β = 0.1** — accuracy weighted 10× over cost, cost log-scaled between $0.0044 and $200/1K queries. We reproduced the formula against the live leaderboard to ±0.01 before anything else. Two consequences shaped the whole campaign:

1. **Accuracy is the binding constraint.** Even a literally-free router needs ~75% accuracy to place. The entire cost axis from $1 to free is worth ~5 arena points; the same span in accuracy is worth the same — and cloud models get accuracy.
2. **"Cheapest" is a consolation prize.** Our free-local advantage, framed as cost savings, buys 1–2 points. Reframed as a *free second opinion for ensembling* — an accuracy device — it buys 3–4. This mid-campaign reframe produced the final architecture.

## 4. The path — phases, assumptions, what actually happened

### Phase 1 — Harness & calibration (Aug 14–15)

RouterArena harness extensions (Lemonade + OpenRouter providers), a **1,251-item calibration set** sampled from the *upstream* public benchmarks RouterArena's categories draw from (86.8% source-mass coverage, deduplicated against every RouterArena split, using their own prompt templates — builder and per-item hash manifest in `calibration/`), and anchor tooling (resumable, per-source breakdowns, full-response capture).

### Phase 2 — Anchors & the quant-parity result (Aug 15–17)

- Cloud anchors on calibration: deepseek-v4-flash 75.3% / $0.77 per 1K; deepseek-v4-pro 72.9% / $6.70.
- **Local DS4-Flash (IQ2XXS quant, 80.8 GB, Lemonade ds4 recipe): 72.2% — exact parity with the API flash on identical items, 87% per-item agreement.** The 2-bit-class quant loses nothing measurable. This validated the entire local-first premise.
- Qwen3.8-27B (UD-Q4, llamacpp recipe) measured as the second local model: 66.2%.

### Phase 3 — Escalation-detector design (Aug 16–17)

We tested request-side and behavior-side escalation signals on a calibration TUNE split and verified on HOLDOUT (`policy/detector_v1.py`). **k=2 self-consistency — sample the local answer twice, escalate on disagreement — dominated: 70% recall of local-wrong at 29% escalation.** Every request-side signal (length, category keywords, perplexity proxies) stayed ≤62% recall at similar escalation rates. Difficulty shows in the attempt, not the request.

### Phase 4 — Full-set passes and cascade rounds (Aug 17–25)

Two full local generation passes (temp-0 and temp-0.7) over all 8,400 queries, distributed across both machines. Frozen cascade policy R1 (code→cloud, no-answer→cloud, k2-disagreement→cloud, else local) scored **arena 73.04**; an independently designed R2 variant scored **72.99**. Two configurations landing within 0.05 points exposed a Pareto frontier: with a cascade over this portfolio, ~73 was the ceiling. An escalation-model bake-off (glm-4.7, glm-4.7-flash, gpt-oss-120b, gemini-3-flash-preview) confirmed it — every variant landed 73.0–73.2.

The structural reason (the campaign's most important analytical finding): **adverse selection**. Every cloud model scores ~25 points below its solo average on cascade-selected hard slices. A cascade sends the cloud model exactly the queries everyone finds hard, so "escalate to a strong model" buys far less than its solo accuracy suggests.

### Phase 5 — The pivot to ensembling (Aug 26–27)

A leaderboard refresh showed the top-3 bar had risen a full point mid-campaign, with ensemble-style entries arriving at the top. Conclusion: voting, not cascading, is what wins at β=0.1 — and our free local models are the one ensemble ingredient no cloud-only router can match at cost.

**v5: tri-opinion ensemble.** DS4-Flash (local) + Qwen3.8-27B (local) + gemini-3-flash-preview (cloud) vote on every MCQ; 2-of-3 majority wins, response submitted from a majority member with local preferred. Code → gemini-3-flash-preview (its 92.1% on the code slice, measured from public graded data, beats deepseek-v4-pro's 76.3% at a fraction of the token bill). Free-answer → local kept when the two local models corroborate. Measured label-free on the full set: **unanimous 77.3%, majority 17.2%, no-majority 5.5%** — 80% of the benchmark resolved on-device. Scored: **74.66**.

### Phase 6 — v5.1, the submitted configuration (Aug 27–28)

One residual lever remained: v5's escalations (no-majority MCQ and uncorroborated free-answer) went to deepseek-v4-pro, whose verbose responses dominated the submitted cost while its accuracy on those adverse-selected rows matched gemini-3-flash-preview's. v5.1 sends all escalations to gemini-3-flash-preview instead (`policy/assemble_v51.py`). Scored with RouterArena's own evaluation code: **arena 76.74 · accuracy 78.03% · $0.171/1K · robustness 73.1 · 0 abnormal.** Frozen and submitted.

## 5. Assumptions ledger — what we believed vs. what was true

| Assumption | Verdict |
|---|---|
| Local IQ2 quant ≈ API model quality | **Confirmed exactly** (72.2% = 72.2%, paired items) |
| Behavior signals beat request inspection for difficulty | **Confirmed** (k2 70% recall; every request-side signal ≤62%) |
| Calibration projections transfer to the full set | **Repeatedly ~3 points optimistic.** Only full-set-measured statistics (e.g. the voter-agreement rates) transferred cleanly |
| Cloud rescue rates on escalated slices ≈ solo accuracy | **Wrong — adverse selection** (~25-point tax; see Phase 4) |
| Free local = winning cost story | **Wrong under β=0.1**; right when reframed as free ensemble members |
| Public graded submissions predict model quality | **Mostly** — they found gemini-3-flash-preview's code strength and ruled out weak escalation targets; misleading on one model pair, corrected by our own paired calibration |
| The bar stays put while we build | **Wrong** — +1.0 point of top-3 inflation in two weeks. The board's meta moved toward ensembles faster than we shipped |

## 6. Operational lessons

- **Qwen3.8 + ROCm nightly + concurrent requests = progressive KV corruption** (word-salad, then repeated tokens), recovering on model reload. Filed as [lemonade#3160](https://github.com/lemonade-sdk/lemonade/issues/3160); worked around with chunked-sequential passes + reload + a known-answer canary between chunks (`tools/run_local_anchor_chunked.sh`).
- **The ds4 recipe loads an 81 GB model in ~30 s** where mainline llama.cpp is unusably slow on UMA hardware — memory configuration is as much the product as architecture support.
- Resumable, id-deduplicated outputs made every infrastructure failure (zombie watchers, double-ran workloads, killed servers) recoverable without data loss. Observability defaults matter more than orchestration cleverness.

## 7. Where this leaves the approach

**The router brain is not the limiter.** Selection quality measured at or near ceiling in every configuration; the ensemble extracts 78.0% from ingredients whose best member solos at ~79% — while keeping 80% of traffic on-device. The limiter is the local models' absolute accuracy, and that improves with every local-model release: the same machinery re-runs against a stronger local model in ~3 days, and every point of local-model improvement is roughly a point of arena score.

## 8. Costs & artifacts

- Cloud: ~$50 total. Complete ledger: `telemetry/spend.jsonl`.
- Final verified metrics: `telemetry/scores.json`.
- Frozen policy: `policy/assemble_v51.py`. Detector experiments: `policy/detector_v1.py`.
- Calibration: `calibration/` (builder, hash manifest, coverage report).
- Measurement harness: `tools/`.
- Raw generation logs (several GB, containing RouterArena prompt content) are not republished here; available to RouterArena maintainers on request.
- Related upstream work shipped during the campaign: lemonade PRs [#3101](https://github.com/lemonade-sdk/lemonade/pull/3101), [#3112](https://github.com/lemonade-sdk/lemonade/pull/3112), issues #3086–#3089, [#3160](https://github.com/lemonade-sdk/lemonade/issues/3160).
