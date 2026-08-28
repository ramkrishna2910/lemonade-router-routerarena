#!/usr/bin/env python3
"""Run one model over the calibration set and grade it.

Usage: run_calibration_anchor.py <model_name> [--split TUNE|HOLDOUT|ALL] [--limit N]

model_name uses the harness convention (fireworks/..., lemonade/...).
Results are incremental and resumable: runs/routerarena/anchors/<safe_name>.jsonl
Grading by answer class:
  MCQ (options present)  boxed/letter extraction, exact letter match
  numeric (GSM8K/MATH/AIME/AsDiv)  boxed/number extraction, numeric equality
  free text (NarrativeQA/WMT/Chess) token-F1 >= 0.5 (approximate; same rule for
    every model, so cross-model deltas are meaningful even if absolute isn't)
  LiveCodeBench  NOT graded here (needs execution); recorded as ungraded
"""

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RA = Path.home() / "work/routerarena"
OUT = Path.home() / "work/router-experiment/runs/routerarena"
sys.path.insert(0, str(RA))

BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
NUMERIC_SOURCES = {"GSM8K", "MATH", "AIME", "AsDiv"}
FREETEXT_SOURCES = {"NarrativeQA", "ChessInstruct"}


def extract_boxed(text):
    matches = BOXED.findall(text or "")
    return matches[-1].strip() if matches else None


def norm_num(text):
    if text is None:
        return None
    cleaned = re.sub(r"[,\$\s]|\\text\{[^}]*\}|\\!", "", str(text))
    cleaned = re.sub(r"\\d?frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", cleaned)
    try:
        if "/" in cleaned:
            num, den = cleaned.split("/", 1)
            return round(float(num) / float(den), 6)
        return round(float(cleaned), 6)
    except (ValueError, ZeroDivisionError):
        return cleaned.lower() or None


def token_f1(a, b):
    ta, tb = set(re.findall(r"\w+", (a or "").lower())), set(re.findall(r"\w+", (b or "").lower()))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    if inter == 0:
        return 0.0
    precision, recall = inter / len(tb), inter / len(ta)
    return 2 * precision * recall / (precision + recall)


def grade_mbpp(item, response):
    import subprocess
    import tempfile

    try:
        payload = json.loads(item["answer"])
    except (json.JSONDecodeError, TypeError):
        return None
    tests = payload.get("mbpp_tests")
    if not tests:
        return None
    m = re.search(r"```(?:python)?\s*\n(.*?)```", response or "", re.DOTALL)
    code = m.group(1) if m else (response or "")
    program = "\n".join(payload.get("imports", [])) + "\n" + code + "\n" + "\n".join(tests)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        proc = subprocess.run(["python3", path], capture_output=True, timeout=10)
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        Path(path).unlink(missing_ok=True)


def grade(item, response):
    if item.get("answer") is None:
        return None  # unlabeled pass (e.g. RouterArena full): responses only
    source = item["ra_source"]
    if source == "LiveCodeBench":
        if item.get("_proxy") == "mbpp" or '"mbpp_tests"' in (item.get("answer") or ""):
            return grade_mbpp(item, response)
        return None  # real LCB needs their execution harness; ungraded here
    if item.get("options"):
        extracted = extract_boxed(response)
        if extracted is None:
            m = re.search(r"answer\s*(?:is|:)\s*\(?([A-Z])\)?", response or "", re.IGNORECASE)
            extracted = m.group(1) if m else None
        if extracted is None:
            return False
        return extracted.strip().upper()[:1] == str(item["answer"]).strip().upper()[:1]
    base = source.split("-")[0].split("_")[0]
    if source in NUMERIC_SOURCES or base in NUMERIC_SOURCES:
        got = norm_num(extract_boxed(response) or (re.findall(r"-?\d[\d,\.\/]*", response or "") or [None])[-1])
        want = norm_num(item["answer"])
        return got is not None and got == want
    if source.startswith("WMT19") or source in FREETEXT_SOURCES:
        return token_f1(response, item["answer"]) >= 0.5
    extracted = extract_boxed(response) or response
    return token_f1(extracted, item["answer"]) >= 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--split", default="ALL", choices=["TUNE", "HOLDOUT", "ALL"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--calibration", default=str(OUT / "calibration.jsonl"))
    ap.add_argument("--store-full", action="store_true",
                    help="store the complete response text (submission/detector inputs)")
    args = ap.parse_args()

    from llm_inference.model_inference import ModelInference

    items = [json.loads(l) for l in Path(args.calibration).open()]
    if args.split != "ALL":
        items = [x for x in items if x["split"] == args.split]
    if args.limit:
        items = items[: args.limit]

    anchors_dir = OUT / "anchors"
    anchors_dir.mkdir(exist_ok=True)
    out_file = anchors_dir / (args.model.replace("/", "_") + ".jsonl")
    done = set()
    if out_file.exists():
        kept = []
        for line in out_file.open():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("success"):
                done.add(row["id"])
                kept.append(line)
        out_file.write_text("".join(kept))  # failed rows retry on resume
    todo = [x for x in items if x["id"] not in done]
    print(f"[anchor] {args.model}: {len(todo)} to run ({len(done)} cached)")

    inference = ModelInference()
    lock = threading.Lock()
    fh = out_file.open("a")

    def one(item):
        result = inference.infer(args.model, item["prompt_formatted"])
        record = {
            "id": item["id"],
            "ra_source": item["ra_source"],
            "split": item["split"],
            "success": result.get("success", False),
            "correct": grade(item, result.get("response")) if result.get("success") else False,
            "tokens": result.get("token_usage"),
            "response_head": (result.get("response") or "")[:400],
        }
        if args.store_full:
            record["response_full"] = result.get("response") or ""
        with lock:
            fh.write(json.dumps(record) + "\n")
            fh.flush()
        return record

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(one, todo))

    fh.close()
    all_records = [json.loads(l) for l in out_file.open()]
    graded = [r for r in all_records if r.get("success") and r["correct"] is not None]
    accuracy = sum(1 for r in graded if r["correct"]) / max(1, len(graded))
    in_tok = sum((r.get("tokens") or {}).get("input_tokens", 0) for r in all_records)
    out_tok = sum((r.get("tokens") or {}).get("output_tokens", 0) for r in all_records)
    print(f"[anchor] {args.model}: acc {accuracy*100:.1f}% on {len(graded)} graded "
          f"({len(all_records)} total) | tokens in/out {in_tok}/{out_tok}")
    by_source = {}
    for r in graded:
        s = by_source.setdefault(r["ra_source"], [0, 0])
        s[1] += 1
        s[0] += 1 if r["correct"] else 0
    summary = {
        "model": args.model,
        "accuracy": accuracy,
        "graded": len(graded),
        "total": len(all_records),
        "tokens": {"input": in_tok, "output": out_tok},
        "by_source": {k: {"acc": v[0] / v[1], "n": v[1]} for k, v in sorted(by_source.items())},
    }
    (anchors_dir / (args.model.replace("/", "_") + ".summary.json")).write_text(
        json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
