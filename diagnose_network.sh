#!/bin/bash
# Network connectivity diagnostic tool for distributed-llama
# Usage: ./diagnose_network.sh <worker_host> <worker_port>

set -e

WORKER_HOST=${1:-"100.78.3.114"}
WORKER_PORT=${2:-"9999"}

echo "🔍 Distributed-Llama Network Diagnostics"
echo "=========================================="
echo ""
echo "Target: $WORKER_HOST:$WORKER_PORT"
echo ""

# Test 1: Basic network connectivity
echo "Test 1: Ping connectivity"
echo "-------------------------"
if ping -c 3 -W 2 "$WORKER_HOST" > /dev/null 2>&1; then
    echo "✅ Host is reachable via ping"
else
    echo "❌ Host is NOT reachable via ping"
    echo "   → Check if host is up and network is configured correctly"
fi
echo ""

# Test 2: TCP port connectivity
echo "Test 2: TCP port connectivity"
echo "-----------------------------"
if timeout 5 bash -c "echo > /dev/tcp/$WORKER_HOST/$WORKER_PORT" 2>/dev/null; then
    echo "✅ Port $WORKER_PORT is OPEN and accepting connections"
else
    echo "❌ Port $WORKER_PORT is NOT accessible"
    echo "   Possible causes:"
    echo "   1. Worker process is not running on $WORKER_HOST"
    echo "   2. Worker is not listening on port $WORKER_PORT"
    echo "   3. Firewall is blocking the connection"
    echo "   4. Worker crashed or failed to start"
    echo ""
    echo "   → Run this command on $WORKER_HOST to check if worker is running:"
    echo "     ps aux | grep 'dllama worker'"
    echo ""
    echo "   → Run this command on $WORKER_HOST to check if port is listening:"
    echo "     netstat -tlnp | grep $WORKER_PORT"
    echo "     # or"
    echo "     ss -tlnp | grep $WORKER_PORT"
fi
echo ""

# Test 3: Check if netcat/nc is available for advanced testing
echo "Test 3: Advanced port scan"
echo "--------------------------"
if command -v nc >/dev/null 2>&1; then
    if timeout 5 nc -zv "$WORKER_HOST" "$WORKER_PORT" 2>&1 | grep -q succeeded; then
        echo "✅ Port scan successful - service is listening"
    else
        echo "❌ Port scan failed - service is not listening"
    fi
elif command -v nmap >/dev/null 2>&1; then
    echo "Running nmap scan..."
    nmap -p "$WORKER_PORT" "$WORKER_HOST" | grep "$WORKER_PORT"
else
    echo "⚠️  Neither nc nor nmap available for advanced testing"
fi
echo ""

# Test 4: DNS resolution
echo "Test 4: DNS resolution"
echo "----------------------"
if host "$WORKER_HOST" > /dev/null 2>&1; then
    echo "✅ Hostname resolves correctly"
    host "$WORKER_HOST"
else
    echo "⚠️  Hostname lookup failed or not a hostname (might be IP)"
fi
echo ""

# Test 5: Check for common firewall issues
echo "Test 5: Local firewall check"
echo "----------------------------"
if command -v iptables >/dev/null 2>&1 && [ "$EUID" -eq 0 ]; then
    echo "Checking iptables rules..."
    if iptables -L OUTPUT -n | grep -q "$WORKER_PORT"; then
        echo "⚠️  Found iptables rules that might affect port $WORKER_PORT"
        iptables -L OUTPUT -n | grep "$WORKER_PORT"
    else
        echo "✅ No blocking OUTPUT rules found"
    fi
elif command -v ufw >/dev/null 2>&1; then
    echo "Checking ufw status..."
    ufw status | grep -i "$WORKER_PORT" || echo "ℹ️  No specific ufw rules for port $WORKER_PORT"
else
    echo "ℹ️  Firewall tools not available or insufficient permissions"
fi
echo ""

# Summary and recommendations
echo "=========================================="
echo "📋 Summary & Recommendations"
echo "=========================================="
echo ""
echo "If port $WORKER_PORT is NOT accessible:"
echo ""
echo "1️⃣  ON THE WORKER NODE ($WORKER_HOST), start the worker:"
echo "   ./dllama worker --port $WORKER_PORT --nthreads 4"
echo ""
echo "2️⃣  Verify worker is running:"
echo "   ps aux | grep 'dllama worker'"
echo ""
echo "3️⃣  Verify port is listening:"
echo "   netstat -tlnp | grep $WORKER_PORT"
echo "   # or"
echo "   ss -tlnp | grep $WORKER_PORT"
echo ""
echo "4️⃣  Check worker logs for errors:"
echo "   # Look for 'Listening on' message"
echo ""
echo "5️⃣  If using firewall, allow the port:"
echo "   # ufw"
echo "   sudo ufw allow $WORKER_PORT/tcp"
echo "   # iptables"
echo "   sudo iptables -A INPUT -p tcp --dport $WORKER_PORT -j ACCEPT"
echo ""
echo "6️⃣  On cloud/VM platforms, check security groups:"
echo "   - AWS: Security Group inbound rules"
echo "   - GCP: Firewall rules"
echo "   - Azure: Network Security Group"
echo ""
