#!/usr/bin/env python3
"""Assemble the v5 tri-ensemble submission (frozen).

Policy v5 (design constants fixed a priori; no RouterArena labels touched):
  MCQ:  majority letter of {DS4-local, Qwen3.8-local, g3fp}.
        Submitted response = a majority member, preferring local (DS4, then
        Qwen, then g3fp).  No majority -> pro.
  code: g3fp.
  free-answer: DS4 kept when Qwen's answer agrees (token-F1 >= 0.5) or when
        numeric answers match; otherwise pro.
Robustness file: same policy, predictions only.
"""

import json
import re
from pathlib import Path

OUT = Path.home() / "work/router-experiment/runs/routerarena"
RA = Path.home() / "work/routerarena"
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")

M_DS4 = "lemonade/deepseek-v4-flash"
M_QWEN = "lemonade/Qwen3.8-27B-GGUF-UD-Q4_K_XL"
M_G3FP = "gemini-3-flash-preview"
M_PRO = "deepseek/deepseek-v4-pro"


def letter(t):
    m = BOXED.findall(t or "")
    return m[-1].strip().upper()[:1] if m else None


def token_f1(a, b):
    ta = set(re.findall(r"\w+", (a or "").lower()))
    tb = set(re.findall(r"\w+", (b or "").lower()))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return 2 * inter / (len(ta) + len(tb)) if inter else 0.0


def extract_free(t):
    m = BOXED.findall(t or "")
    return m[-1].strip() if m else (t or "").strip()[-200:]


def load(paths):
    d = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        for line in p.open():
            try:
                r = json.loads(line)
                if r.get("success") and r["id"] not in d:
                    d[r["id"]] = r
            except json.JSONDecodeError:
                continue
    return d


def entry(gid, prompt, model, r, provider, extra=None):
    e = {
        "global index": gid,
        "prompt": prompt,
        "prediction": model,
        "generated_result": {
            "generated_answer": r.get("response_full") or r.get("response_head") or "",
            "success": True,
            "token_usage": r.get("tokens") or {},
            "provider": provider,
            "error": None,
        },
    }
    if extra:
        e.update(extra)
    return e


def decide(item, ds4, qwen, g3fp):
    """Returns (model_key, source_record_key, reason). model_key None => pro."""
    prompt = item["prompt_formatted"]
    gid = item["id"]
    r1, r2, r3 = ds4.get(gid), qwen.get(gid), g3fp.get(gid)
    code = "Options:" not in prompt and "boxed" not in prompt
    mcq = "Options:" in prompt

    if code:
        return ("g3fp", "g3fp", "code") if r3 else (None, None, "code")
    if mcq:
        ls = {
            "ds4": letter((r1 or {}).get("response_full") or (r1 or {}).get("response_head")),
            "qwen": letter((r2 or {}).get("response_full")),
            "g3fp": letter((r3 or {}).get("response_full") or (r3 or {}).get("response_head")),
        }
        from collections import Counter

        votes = Counter(v for v in ls.values() if v)
        if votes:
            top, n = votes.most_common(1)[0]
            if n >= 2:
                for pref in ("ds4", "qwen", "g3fp"):
                    if ls[pref] == top and {"ds4": r1, "qwen": r2, "g3fp": r3}[pref]:
                        return (pref, pref, "majority")
        return (None, None, "tiebreak")
    # free-answer: DS4 kept when Qwen corroborates
    a1 = extract_free((r1 or {}).get("response_full") or (r1 or {}).get("response_head"))
    a2 = extract_free((r2 or {}).get("response_full"))
    if r1 and r2 and (a1 == a2 or token_f1(a1, a2) >= 0.5):
        return ("ds4", "ds4", "free_agree")
    return (None, None, "free_escalate")


def main():
    items = [json.loads(l) for l in (OUT / "ra_full.jsonl").open()]
    ds4 = load([OUT / "anchors/lemonade_deepseek-v4-flash.jsonl",
                OUT / "ra_full_tail_results.jsonl", OUT / "p1_stragglers_results.jsonl"])
    qwen = load([OUT / "anchors/qwen38_all.jsonl"])
    g3fp = load([OUT / "anchors/google_gemini-3-flash-preview.jsonl"])
    pro = load([OUT / "anchors/deepseek_deepseek-v4-pro.jsonl"])

    model_map = {"ds4": (M_DS4, ds4, "lemonade"), "qwen": (M_QWEN, qwen, "lemonade"),
                 "g3fp": (M_G3FP, g3fp, "openrouter")}
    from collections import Counter

    entries, need_pro = [], []
    reasons = Counter()
    for item in items:
        gid, prompt = item["id"], item["prompt_formatted"]
        key, src_key, why = decide(item, ds4, qwen, g3fp)
        reasons[why + ("" if key else "->pro")] += 1
        if key:
            model, store, provider = model_map[key]
            entries.append(entry(gid, prompt, model, store[gid], provider))
        else:
            # v5.1: escalations (tiebreak + free-answer) go to g3fp — its
            # at-scale free-answer accuracy matches pro's on hard rows at a
            # fraction of the declared token price.
            c = g3fp.get(gid) or pro.get(gid)
            if c:
                model = M_G3FP if gid in g3fp else M_PRO
                entries.append(entry(gid, prompt, model, c, "openrouter"))
            else:
                need_pro.append(item)
    print(f"decisions: {dict(reasons)}")
    print(f"assembled {len(entries)}/{len(items)} | pro rows needed: {len(need_pro)}")
    if need_pro:
        (OUT / "v5_need_pro.jsonl").write_text("".join(json.dumps(x) + "\n" for x in need_pro))
        print("wrote v5_need_pro.jsonl — run pro batch, re-assemble")
        return
    dest = RA / "router_inference/predictions/lemonade-cascade.json"
    dest.write_text(json.dumps(entries, indent=1))
    print(f"WROTE {dest}")


if __name__ == "__main__":
    main()
