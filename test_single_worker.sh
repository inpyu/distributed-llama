#!/bin/bash
# Minimal test: 2 nodes (root + 1 worker)

set -e

cd "$( dirname "${BASH_SOURCE[0]}" )"

pkill -f "dllama worker" || true
sleep 1

echo "🚀 Starting single worker..."
./dllama worker --port 10001 --nthreads 4 > worker1.log 2>&1 &
WORKER_PID=$!
echo "  Worker started on port 10001 (PID: $WORKER_PID)"

sleep 2

if ! kill -0 $WORKER_PID 2>/dev/null; then
    echo "❌ Worker failed to start"
    cat worker1.log
    exit 1
fi

echo "✅ Worker running"
echo ""
echo "📊 Running inference on 2 nodes..."
echo ""

timeout 60 ./dllama inference \
    --model dllama_model_original_q40.m \
    --tokenizer dllama_tokenizer_llama3_8B.t \
    --buffer-float-type q80 \
    --prompt "Hi" \
    --steps 8 \
    --nthreads 4 \
    --collective star \
    --workers localhost:10001 2>&1 | tee root_output.log

EXIT_CODE=$?

echo ""
echo "🛑 Stopping worker..."
kill $WORKER_PID 2>/dev/null || true
sleep 1

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Test successful"
else
    echo "❌ Test failed (exit: $EXIT_CODE)"
    echo ""
    echo "=== Worker log ==="
    cat worker1.log
fi

exit $EXIT_CODE
