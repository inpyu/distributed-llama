#!/bin/bash
# Test connection timeout functionality
# This script simulates connection to a non-existent worker to verify timeout works

echo "🧪 Testing connection timeout functionality..."
echo ""
echo "This test will attempt to connect to a non-responsive worker"
echo "and verify that the connection times out properly (30 seconds)."
echo ""
echo "Expected behavior:"
echo "  - Connection attempt message"
echo "  - Timeout message after ~30 seconds"
echo "  - Helpful troubleshooting steps"
echo ""
echo "Press Ctrl+C if you want to abort early."
echo ""
echo "Starting test in 3 seconds..."
sleep 3

# Use a non-routable IP address (192.0.2.1 is TEST-NET-1, reserved for documentation)
# This ensures the connection will timeout rather than fail immediately
FAKE_WORKER="192.0.2.1:9999"

echo "Attempting to connect to unreachable worker: $FAKE_WORKER"
echo ""

START_TIME=$(date +%s)

# This should timeout after 30 seconds
timeout 35 ./dllama inference \
    --model dllama_model_original_q40.m \
    --tokenizer dllama_tokenizer_llama3_8B.t \
    --workers "$FAKE_WORKER" \
    --prompt "test" \
    --steps 1 \
    --nthreads 2 2>&1 | tee timeout_test.log || true

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "Test completed in ${ELAPSED} seconds"
echo ""

if [ $ELAPSED -ge 30 ] && [ $ELAPSED -le 35 ]; then
    echo "✅ Timeout test PASSED"
    echo "   Connection timed out as expected (~30 seconds)"
    if grep -q "Connection timeout" timeout_test.log; then
        echo "✅ Timeout message displayed correctly"
    fi
    if grep -q "Troubleshooting steps" timeout_test.log; then
        echo "✅ Troubleshooting help displayed"
    fi
else
    echo "⚠️  Unexpected timing"
    echo "   Expected: 30-35 seconds"
    echo "   Actual: ${ELAPSED} seconds"
fi

echo ""
echo "Log file saved to: timeout_test.log"
