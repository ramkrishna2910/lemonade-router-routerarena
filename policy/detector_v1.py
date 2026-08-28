#!/usr/bin/env python3
"""Escalation-detector v1: rank single-response signals from the local DS4
attempt against ground truth on calibration TUNE; verify chosen config on
HOLDOUT. Signals must be computable at serving time from the local response
(and the prompt), with no labels.

Data: anchors/ds4_local_calibsample.jsonl (response_head, tokens, correct)
joined with calibration_sample300.jsonl (options, split, prompt).
Cloud outcomes: anchors/fireworks_deepseek-v4-pro.jsonl (escalation target
via OpenRouter at submission; Fireworks per-item outcomes are the stand-in —
same model family/vintage).
"""

import json
import re
from pathlib import Path

OUT = Path.home() / "work/router-experiment/runs/routerarena"
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
HEDGES = re.compile(
    r"\b(likely|approximately|closest|probably|assum\w+|uncertain|not sure|"
    r"however|but wait|alternatively|hard to|cannot determine)\b", re.IGNORECASE)

cal = {x["id"]: x for x in (json.loads(l) for l in (OUT / "calibration_sample300.jsonl").open())}
pro = {}
for l in (OUT / "anchors/fireworks_deepseek-v4-pro.jsonl").open():
    r = json.loads(l)
    if r.get("success") and r.get("correct") is not None:
        pro[r["id"]] = r["correct"]

rows = []
for l in (OUT / "anchors/ds4_local_calibsample.jsonl").open():
    r = json.loads(l)
    item = cal.get(r["id"])
    if not item or not r.get("success") or r.get("correct") is None:
        continue
    head = r.get("response_head") or ""
    out_tok = (r.get("tokens") or {}).get("output_tokens", 0)
    prompt = item.get("prompt_formatted") or ""
    boxed = BOXED.findall(head)
    letter = boxed[-1].strip().upper()[:1] if boxed else None
    n_opts = len(item.get("options") or [])
    rows.append({
        "id": r["id"],
        "split": item["split"],
        "correct": bool(r["correct"]),
        "pro_correct": pro.get(r["id"]),
        # --- signals ---
        "s_no_boxed": (not boxed) and out_tok < 380,       # truncation-safe
        "s_long": out_tok > 1200,
        "s_hedge": bool(HEDGES.search(head)),
        "s_bad_letter": bool(letter) and n_opts > 0 and not (
            letter.isalpha() and ord(letter) - 65 < n_opts),
        "s_freetext_prompt": n_opts == 0,                  # request-derived
    })

SIGNALS = ["s_no_boxed", "s_long", "s_hedge", "s_bad_letter", "s_freetext_prompt"]


def evaluate(subset, esc_fn, name):
    esc = [r for r in subset if esc_fn(r)]
    keep = [r for r in subset if not esc_fn(r)]
    wrong = [r for r in subset if not r["correct"]]
    caught = [r for r in esc if not r["correct"]]
    # cascade accuracy: kept locals as-is; escalated take pro's outcome
    acc = (sum(r["correct"] for r in keep) +
           sum(1 for r in esc if r["pro_correct"]))
    n = len(subset)
    print(f"  {name:34s} esc {len(esc)/n*100:4.1f}% | recall(local-wrong) "
          f"{len(caught)/max(1,len(wrong))*100:4.1f}% | cascade acc {acc/n*100:.1f}%")
    return acc / n


tune = [r for r in rows if r["split"] == "TUNE"]
hold = [r for r in rows if r["split"] == "HOLDOUT"]
base_t = sum(r["correct"] for r in tune) / len(tune)
print(f"TUNE n={len(tune)} local-only {base_t*100:.1f}% | "
      f"pro-only {sum(bool(r['pro_correct']) for r in tune)/len(tune)*100:.1f}%")
print("single signals on TUNE:")
for s in SIGNALS:
    evaluate(tune, lambda r, s=s: r[s], s)
print("combos on TUNE:")
combos = {
    "no_boxed|freetext": lambda r: r["s_no_boxed"] or r["s_freetext_prompt"],
    "no_boxed|freetext|hedge": lambda r: r["s_no_boxed"] or r["s_freetext_prompt"] or r["s_hedge"],
    "no_boxed|freetext|long": lambda r: r["s_no_boxed"] or r["s_freetext_prompt"] or r["s_long"],
    "any-signal": lambda r: any(r[s] for s in SIGNALS),
}
best_name, best_acc = None, 0
for name, fn in combos.items():
    a = evaluate(tune, fn, name)
    if a > best_acc:
        best_name, best_acc = name, a
print(f"\nchosen on TUNE: {best_name} -> verify on HOLDOUT n={len(hold)}:")
print(f"  local-only {sum(r['correct'] for r in hold)/len(hold)*100:.1f}%")
evaluate(hold, combos[best_name], best_name)
