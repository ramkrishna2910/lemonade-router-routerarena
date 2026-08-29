# lemonade-liquid-router — RouterArena submission methodology

Supporting artifacts for the **lemonade-liquid-router** submission to [RouterArena](https://github.com/RouteWorks/RouterArena): a local-first router built on three on-device models (two generator voters and an embedding referee), all served by a single instance of [Lemonade](https://github.com/lemonade-sdk/lemonade), AMD's open-source local LLM server, plus one cloud voter. **~80% of the benchmark is answered entirely on-device.**

**Final verified scores** (RouterArena's own evaluation code, full 8,400-query split): accuracy **78.10%** · **$0.178/1K queries** · arena score (β=0.10) **76.75** · robustness **73.6** · 0 abnormal entries.

## Contents

| Path | What it is |
|---|---|
| [`METHODOLOGY.md`](METHODOLOGY.md) | The submitted policy, the provenance of every design constant, and how each claim can be checked |
| [`policy/assemble_v52.py`](policy/assemble_v52.py) | The frozen routing policy, exactly as run to produce the submitted predictions |
| [`calibration/build_calibration.py`](calibration/build_calibration.py) | Builder for the 1,251-item external calibration set (sampled from the upstream public benchmarks, deduplicated against every RouterArena split) |
| [`calibration/calibration_manifest.jsonl`](calibration/calibration_manifest.jsonl) | Per-item source + SHA-256 hashes — verify the dedup claim without us redistributing benchmark content |
| [`calibration/coverage_report.json`](calibration/coverage_report.json) | Source-mass coverage of the calibration set (86.8%) |
| [`telemetry/scores.json`](telemetry/scores.json) | Final verified metrics |
| [`tools/run_calibration_anchor.py`](tools/run_calibration_anchor.py) | The harness behind every calibration measurement cited in the methodology |

## Compliance in one paragraph

RouterArena is evaluation-only: no router component may be trained or tuned on its data or labels. Every threshold and design choice in this submission is justified only by (i) the external calibration set in this repo, (ii) our own runtime/token telemetry, or (iii) aggregate statistics from the graded submissions the RouterArena maintainers publish in their public repo — the same information available to every submitter, used only for model selection (which model fills a slot), never to fit any per-query component. Full-set labels were touched only by pre-registered scoring runs of frozen configurations. The calibration manifest exists so this claim is checkable, item by item.

## Replication

The submission PR against RouterArena contains the complete recipe: serve `deepseek-v4-flash` (ds4 recipe) and `Qwen3.8-27B-GGUF-UD-Q4_K_XL` (llamacpp recipe) with the public Lemonade release, set `OPENROUTER_API_KEY`, and run the standard RouterArena pipeline with router name `lemonade-liquid-router`. The local side is deterministic (temperature 0, thinking disabled); the local IQ2XXS DeepSeek-V4-Flash quant scores identically to the API model on the calibration set.

