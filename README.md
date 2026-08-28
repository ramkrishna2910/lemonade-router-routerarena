# lemonade-router — RouterArena methodology & artifacts

Public methodology repo for the **lemonade-router** submission to
[RouterArena](https://github.com/RouteWorks/RouterArena): a local-first tri-ensemble router in which two of three voters run on the submitter's own hardware via [Lemonade](https://github.com/lemonade-sdk/lemonade), AMD's open-source local LLM server. **~80% of the benchmark is answered entirely on-device.**

**Final verified scores** (RouterArena's own evaluation code, full 8,400-query split): accuracy **78.03%**, cost **$0.171/1K queries**, arena score (β=0.10) **76.74**, robustness **73.1**, 0 abnormal entries. See [`telemetry/scores.json`](telemetry/scores.json).

## What's here

| Path | Contents |
|---|---|
| [`METHODOLOGY.md`](METHODOLOGY.md) | The full campaign report: goal, ground rules, phases, assumptions ledger, and what was actually measured |
| [`policy/assemble_v51.py`](policy/assemble_v51.py) | The frozen v5.1 routing policy, exactly as run to produce the submitted predictions |
| [`policy/detector_v1.py`](policy/detector_v1.py) | The escalation-signal experiments (self-consistency, request-side signals) that informed the design |
| [`calibration/build_calibration.py`](calibration/build_calibration.py) | Builder for the 1,251-item external calibration set, sampled from the upstream public benchmarks RouterArena's categories draw from |
| [`calibration/calibration_manifest.jsonl`](calibration/calibration_manifest.jsonl) | Per-item manifest (source + SHA-256 of question and formatted prompt) so anyone can verify the set is deduplicated against every RouterArena split without us redistributing benchmark content |
| [`calibration/coverage_report.json`](calibration/coverage_report.json) | Source-mass coverage of the calibration set (86.8%) |
| [`telemetry/spend.jsonl`](telemetry/spend.jsonl) | The complete cloud-spend ledger for the campaign (~$50 total) |
| [`telemetry/scores.json`](telemetry/scores.json) | Final verified metrics |
| [`tools/`](tools/) | The anchor/measurement harness: resumable runners for calibration and full-set passes, including the chunked-sequential workaround for a local serving bug we filed upstream ([lemonade#3160](https://github.com/lemonade-sdk/lemonade/issues/3160)) |

## Compliance in one paragraph

RouterArena is evaluation-only: no router component may be trained or tuned on its data or labels. Every threshold and design choice in this submission is justified only by (i) the external calibration set in this repo, (ii) our own runtime/token telemetry, or (iii) public graded submissions already in the RouterArena repo — the same information available to every submitter. Full-set labels were touched only by pre-registered scoring runs of frozen configurations. The calibration manifest exists so this claim is checkable.

## Replication

The submission PR against RouterArena contains the complete replication recipe (candidate models, routing policy, and the standard RouterArena pipeline commands). Short version: serve `deepseek-v4-flash` (ds4 recipe) and `Qwen3.8-27B-GGUF-UD-Q4_K_XL` (llamacpp recipe) with the public Lemonade release, set `OPENROUTER_API_KEY`, and run the RouterArena pipeline with router name `lemonade-router`. The local side is deterministic (temperature 0, thinking disabled); we verified the local IQ2XXS DeepSeek-V4-Flash quant scores identically to the API model on the calibration set.

Raw generation logs (all local and cloud passes, full responses) total several GB and contain RouterArena prompt content, so they are not republished here; they are available to RouterArena maintainers on request: ramkrishna2910@gmail.com.
