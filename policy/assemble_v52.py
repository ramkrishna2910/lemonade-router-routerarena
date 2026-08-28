#!/usr/bin/env python3
"""Assemble the v5.2 submission (frozen).

v5.2 = v5.1 + embedding-corroboration veto on free-answer FALLBACK rows only:
rows where >=1 side produced no boxed answer, so token-F1 compared degraded
last-200-chars extracts. There, keeping the local answer additionally requires
LFM2.5-Embedding-350M (F16 GGUF, served by lemonade llamacpp) cosine >= 0.7
between the two extracted answers. Threshold chosen label-free (g3fp-agreement
pseudo-reference; see LFM_JUDGE_EXPERIMENT.md). Boxed-both rows, MCQ, and code
are unchanged from v5.1.
"""

import json
import math
import re
import urllib.request
from collections import Counter
from pathlib import Path

OUT = Path.home() / "work/router-experiment/runs/routerarena"
RA = Path.home() / "work/routerarena"
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
EMB_MODEL = "user.LFM2.5-Embedding-350M"
EMB_URL = "http://localhost:8000/api/v1/embeddings"
EMB_THRESHOLD = 0.7

M_DS4 = "lemonade/deepseek-v4-flash"
M_QWEN = "lemonade/Qwen3.8-27B-GGUF-UD-Q4_K_XL"
M_G3FP = "gemini-3-flash-preview"


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
    return (m[-1].strip() if m else (t or "").strip()[-200:]), bool(m)


def embed_cos_batch(pairs):
    """pairs: list of (a1, a2) -> list of cosines, via lemonade embeddings."""
    cosines = []
    B = 16
    for i in range(0, len(pairs), B):
        chunk = pairs[i:i + B]
        texts = [t for p in chunk for t in p]
        req = urllib.request.Request(
            EMB_URL,
            data=json.dumps({"model": EMB_MODEL, "input": texts}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as f:
            vecs = [e["embedding"] for e in json.load(f)["data"]]
        for j in range(len(chunk)):
            a, b = vecs[2 * j], vecs[2 * j + 1]
            num = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            cosines.append(num / (na * nb) if na and nb else 0.0)
    return cosines


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


def resp(r):
    return (r or {}).get("response_full") or (r or {}).get("response_head") or ""


def entry(gid, prompt, model, r, provider):
    return {
        "global index": gid,
        "prompt": prompt,
        "prediction": model,
        "generated_result": {
            "generated_answer": resp(r),
            "success": True,
            "token_usage": r.get("tokens") or {},
            "provider": provider,
            "error": None,
        },
    }


def decide(item, ds4, qwen, g3fp, cos_lookup):
    prompt, gid = item["prompt_formatted"], item["id"]
    r1, r2, r3 = ds4.get(gid), qwen.get(gid), g3fp.get(gid)
    if "Options:" not in prompt and "boxed" not in prompt:
        return "g3fp", "code"
    if "Options:" in prompt:
        ls = {"ds4": letter(resp(r1)), "qwen": letter(resp(r2)), "g3fp": letter(resp(r3))}
        votes = Counter(v for v in ls.values() if v)
        if votes:
            top, n = votes.most_common(1)[0]
            if n >= 2:
                for pref in ("ds4", "qwen", "g3fp"):
                    if ls[pref] == top and {"ds4": r1, "qwen": r2, "g3fp": r3}[pref]:
                        return pref, "majority"
        return "g3fp", "tiebreak"
    a1, b1 = extract_free(resp(r1))
    a2, b2 = extract_free(resp(r2))
    f1_keep = bool(r1 and r2 and a1 and a2 and (a1 == a2 or token_f1(a1, a2) >= 0.5))
    if not f1_keep:
        return "g3fp", "free_escalate"
    if b1 and b2:
        return "ds4", "free_agree"
    # fallback row: embedding veto
    c = cos_lookup(gid, a1, a2)
    if c >= EMB_THRESHOLD:
        return "ds4", "free_agree_emb"
    return "g3fp", "free_emb_veto"


def assemble(items, ds4, qwen, g3fp, cos_lookup, dest):
    stores = {"ds4": (M_DS4, ds4, "lemonade"), "qwen": (M_QWEN, qwen, "lemonade"),
              "g3fp": (M_G3FP, g3fp, "openrouter")}
    entries, reasons = [], Counter()
    for item in items:
        key, why = decide(item, ds4, qwen, g3fp, cos_lookup)
        reasons[why] += 1
        model, store, provider = stores[key]
        r = store.get(item["id"])
        assert r, f"missing {key} row for {item['id']}"
        entries.append(entry(item["id"], item["prompt_formatted"], model, r, provider))
    print(f"{dest.name}: {dict(reasons)}")
    dest.write_text(json.dumps(entries, indent=1))
    print(f"WROTE {dest} ({len(entries)} rows)")


def main():
    # full set: reuse the lemonade-served cosines measured for the experiment
    pre = {r["id"]: r["lemonade_emb_cos"]
           for r in json.load((OUT / "lfm_emb_cosines_full.json").open())
           if "lemonade_emb_cos" in r}

    def full_cos(gid, a1, a2):
        if gid in pre:
            return pre[gid]
        return embed_cos_batch([(a1, a2)])[0]

    items = [json.loads(l) for l in (OUT / "ra_full.jsonl").open()]
    ds4 = load([OUT / "anchors/lemonade_deepseek-v4-flash.jsonl",
                OUT / "ra_full_tail_results.jsonl", OUT / "p1_stragglers_results.jsonl"])
    qwen = load([OUT / "anchors/qwen38_all.jsonl"])
    g3fp = load([OUT / "anchors/google_gemini-3-flash-preview.jsonl"])
    assemble(items, ds4, qwen, g3fp, full_cos,
             RA / "router_inference/predictions/lemonade-router.json")

    # robustness: reproduce the v5.1 rob assembly exactly (transcript-recovered:
    # dataset/router_robustness.json rows, robq-/rob- id-prefix stripping,
    # perturbed-prompt classification), then apply the same embedding veto.
    def load_strip(f, strip=None):
        d = {}
        for l in (OUT / f).open():
            try:
                r = json.loads(l)
                if r.get("success"):
                    k = r["id"]
                    if strip and k.startswith(strip):
                        k = k[len(strip):]
                    d[k] = r
            except json.JSONDecodeError:
                continue
        return d

    rob_ids = {json.loads(l)["id"] for l in (OUT / "ra_robustness.jsonl").open()}
    rob_ds4 = load_strip("rob_p1_results.jsonl")
    rob_qwen = load_strip("anchors/qwen38_robustness.jsonl", strip="robq-")
    rob_g3fp = {k: v for k, v in
                load_strip("anchors/google_gemini-3-flash-preview.jsonl", strip="rob-").items()
                if k in rob_ids}
    rob = json.loads((RA / "dataset/router_robustness.json").read_text())

    rob_entries, reasons = [], Counter()
    for row in rob:
        gid, prompt = row["global index"], row["prompt_formatted"]
        item = {"id": gid, "prompt_formatted": prompt}
        key, why = decide(item, rob_ds4, rob_qwen, rob_g3fp,
                          lambda g, a1, a2: embed_cos_batch([(a1, a2)])[0])
        reasons[why] += 1
        rob_entries.append({"global index": gid, "prompt": prompt,
                            "prediction": {"ds4": M_DS4, "qwen": M_QWEN, "g3fp": M_G3FP}[key]})
    print(f"robustness: {dict(reasons)}")
    dest = RA / "router_inference/predictions/lemonade-router-robustness.json"
    dest.write_text(json.dumps(rob_entries, indent=1))
    print(f"WROTE {dest} ({len(rob_entries)} rows)")


if __name__ == "__main__":
    main()
