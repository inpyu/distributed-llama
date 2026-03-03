#!/bin/bash
# Local 4-node test for distributed-llama
# Usage: ./test_local_4nodes.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Kill any existing worker processes
pkill -f "dllama worker" || true
sleep 1

# Start 3 worker nodes in background
echo "🚀 Starting worker nodes..."

./dllama worker --port 10001 --nthreads 4 > worker1.log 2>&1 &
WORKER1_PID=$!
echo "  Worker 1 started on port 10001 (PID: $WORKER1_PID)"

./dllama worker --port 10002 --nthreads 4 > worker2.log 2>&1 &
WORKER2_PID=$!
echo "  Worker 2 started on port 10002 (PID: $WORKER2_PID)"

./dllama worker --port 10003 --nthreads 4 > worker3.log 2>&1 &
WORKER3_PID=$!
echo "  Worker 3 started on port 10003 (PID: $WORKER3_PID)"

# Wait for workers to start
echo "⏳ Waiting 2 seconds for workers to initialize..."
sleep 2

# Check if workers are running
for pid in $WORKER1_PID $WORKER2_PID $WORKER3_PID; do
    if ! kill -0 $pid 2>/dev/null; then
        echo "❌ Worker with PID $pid failed to start"
        exit 1
    fi
done

echo "✅ All workers started successfully"
echo ""
echo "📊 Running inference on 4 nodes (1 root + 3 workers)..."
echo ""

# Run root inference
./dllama inference \
    --model dllama_model_original_q40.m \
    --tokenizer dllama_tokenizer_llama3_8B.t \
    --buffer-float-type q80 \
    --prompt "Hello world" \
    --steps 16 \
    --nthreads 4 \
    --collective auto \
    --workers localhost:10001 localhost:10002 localhost:10003

INFERENCE_EXIT=$?

echo ""
echo "🛑 Stopping worker nodes..."
kill $WORKER1_PID $WORKER2_PID $WORKER3_PID 2>/dev/null || true
sleep 1

if [ $INFERENCE_EXIT -eq 0 ]; then
    echo "✅ Test completed successfully"
else
    echo "❌ Test failed with exit code $INFERENCE_EXIT"
    echo ""
    echo "Worker logs:"
    echo "--- Worker 1 ---"
    tail -20 worker1.log
    echo "--- Worker 2 ---"
    tail -20 worker2.log
    echo "--- Worker 3 ---"
    tail -20 worker3.log
fi

exit $INFERENCE_EXIT
