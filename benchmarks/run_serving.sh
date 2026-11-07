#!/usr/bin/env bash
# Online serving sweep: start a server, run bench_serving.py at several request rates, stop it.
#   benchmarks/run_serving.sh tinyserve Qwen/Qwen3-0.6B
#   benchmarks/run_serving.sh vllm Qwen/Qwen3-0.6B          (inside a venv with vllm)
set -euo pipefail
BACKEND=${1:-tinyserve}
MODEL=${2:-Qwen/Qwen3-0.6B}
RATES=${RATES:-"4 8 16 32"}
PORT=${PORT:-8011}
OUT=${OUT:-benchmarks/results/serving.jsonl}
cd "$(dirname "$0")/.."
mkdir -p "$(dirname "$OUT")"

if [ "$BACKEND" = vllm ]; then
  vllm serve "$MODEL" --port "$PORT" --max-model-len 4096 --gpu-memory-utilization 0.85 --max-num-seqs 128 \
    > /tmp/serve_$BACKEND.log 2>&1 &
else
  python -m tinyserve.server --model "$MODEL" --port "$PORT" --max-num-seqs 128 > /tmp/serve_$BACKEND.log 2>&1 &
fi
PID=$!
trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null || true' EXIT
until curl -sf "localhost:$PORT/v1/models" > /dev/null; do
  kill -0 $PID 2>/dev/null || { echo "server died"; tail -20 /tmp/serve_$BACKEND.log; exit 1; }
  sleep 1
done
for r in $RATES; do
  python benchmarks/bench_serving.py --base-url "http://localhost:$PORT" --model "$MODEL" --label "$BACKEND" \
    --request-rate "$r" --num-prompts 200 --out "$OUT"
done
