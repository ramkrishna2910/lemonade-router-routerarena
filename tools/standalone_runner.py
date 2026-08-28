#!/usr/bin/env python3
"""Standalone (stdlib-only) runner for ds4-server: reads items jsonl, posts
chat completions, appends results jsonl with full responses. Resumable.

Usage: tail_runner.py <items.jsonl> <out.jsonl> [--temp T] [--workers N]
"""
import json
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

items_path, out_path = sys.argv[1], sys.argv[2]
temp = float(sys.argv[sys.argv.index("--temp") + 1]) if "--temp" in sys.argv else 0.0
workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
URL = sys.argv[sys.argv.index("--url") + 1] if "--url" in sys.argv else "http://127.0.0.1:18200/v1/chat/completions"
MODEL = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "deepseek-v4-flash"
NO_THINK = "--no-think" in sys.argv
CHUNK = int(sys.argv[sys.argv.index("--chunk-limit") + 1]) if "--chunk-limit" in sys.argv else 0

items = [json.loads(l) for l in open(items_path)]
done = set()
try:
    for l in open(out_path):
        try:
            done.add(json.loads(l)["id"])
        except json.JSONDecodeError:
            pass
except FileNotFoundError:
    pass
todo = [x for x in items if x["id"] not in done]
if CHUNK:
    todo = todo[:CHUNK]
print(f"[tail] {len(todo)} to run ({len(done)} cached)", flush=True)

lock = threading.Lock()
fh = open(out_path, "a")

def one(item):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": item["prompt_formatted"]}],
        "max_tokens": 4096,
        "temperature": temp,
    }
    if NO_THINK:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    record = {"id": item["id"], "ra_source": item.get("ra_source"), "success": False}
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.load(r)
        record["success"] = True
        record["response_full"] = d["choices"][0]["message"]["content"] or ""
        record["tokens"] = {
            "input_tokens": d.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": d.get("usage", {}).get("completion_tokens", 0),
        }
    except Exception as e:
        record["error"] = str(e)[:200]
    with lock:
        fh.write(json.dumps(record) + "\n")
        fh.flush()

with ThreadPoolExecutor(max_workers=workers) as pool:
    list(pool.map(one, todo))
fh.close()
n_ok = sum(1 for l in open(out_path) if json.loads(l).get("success"))
print(f"[tail] done: {n_ok} ok / {sum(1 for _ in open(out_path))} rows", flush=True)
