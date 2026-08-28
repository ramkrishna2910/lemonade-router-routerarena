#!/usr/bin/env bash
# Chunked sequential local anchor with reload + canary between chunks.
# Works around llama-server state corruption on new-arch models: each chunk
# gets a fresh model load; a canary probe validates the chunk; a failed canary
# invalidates the chunk's rows (they retry next pass).
set -u
MODEL="lemonade/Qwen3.8-27B-GGUF-UD-Q4_K_XL"
REG_NAME="Qwen3.8-27B-GGUF-UD-Q4_K_XL"
CHUNK=150
TOTAL=1251
RA=~/work/routerarena
OUT=~/work/router-experiment/runs/routerarena/anchors
JSONL="$OUT/lemonade_Qwen3.8-27B-GGUF-UD-Q4_K_XL.jsonl"
BASE=http://localhost:8000/api/v1

canary_ok() {
  RESP=$(curl -s -m 300 -X POST "$BASE/chat/completions" -H "Content-Type: application/json" \
    -d '{"model":"'"$REG_NAME"'","messages":[{"role":"user","content":"What is 17+25? Reply with only the number."}],"max_tokens":30,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'] or '')" 2>/dev/null)
  echo "$RESP" | grep -q "42"
}

cd "$RA" && set -a && . ./.env && set +a
pass=0
while true; do
  pass=$((pass+1))
  done_rows=$(wc -l < "$JSONL" 2>/dev/null || echo 0)
  if [ "$done_rows" -ge "$TOTAL" ]; then echo "[chunked] complete: $done_rows rows"; break; fi
  if [ "$pass" -gt 30 ]; then echo "[chunked] too many passes, aborting"; exit 1; fi
  echo "[chunked] pass $pass: $done_rows done, reloading model..."
  curl -s -X POST "$BASE/unload" -H "Content-Type: application/json" -d '{"model_name":"'"$REG_NAME"'"}' >/dev/null
  curl -s -m 600 -X POST "$BASE/load" -H "Content-Type: application/json" -d '{"model_name":"'"$REG_NAME"'"}' >/dev/null
  canary_ok || { echo "[chunked] canary failed after fresh load — aborting"; exit 1; }
  limit=$((done_rows + CHUNK)); [ "$limit" -gt "$TOTAL" ] && limit=$TOTAL
  before=$(wc -l < "$JSONL" 2>/dev/null || echo 0)
  ~/.local/bin/uv run python ~/work/router-experiment/analysis/routerarena/run_calibration_anchor.py \
    "$MODEL" --workers 1 --limit "$limit" >> /tmp/chunked-anchor.log 2>&1
  if canary_ok; then
    echo "[chunked] pass $pass ok ($(wc -l < "$JSONL") rows)"
  else
    echo "[chunked] canary FAILED after pass $pass — invalidating chunk"
    head -n "$before" "$JSONL" > "$JSONL.tmp" && mv "$JSONL.tmp" "$JSONL"
  fi
done
echo "[chunked] running final summary"
~/.local/bin/uv run python ~/work/router-experiment/analysis/routerarena/run_calibration_anchor.py "$MODEL" --workers 1 >> /tmp/chunked-anchor.log 2>&1
