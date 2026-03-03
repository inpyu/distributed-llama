# Troubleshooting Guide

## Network Connection Issues

### Symptom: Connection hangs at "connecting to worker"

```
⭕ Socket[0]: connecting to 100.78.3.114:9999 worker
```

#### Quick Diagnosis

Run the diagnostic script:

```bash
./diagnose_network.sh 100.78.3.114 9999
```

#### Common Causes and Solutions

##### 1. Worker Not Running

**Symptom**: Port is not accessible

**Solution**: Start the worker on the remote node

```bash
# On the worker node (100.78.3.114)
./dllama worker --port 9999 --nthreads 4
```

Verify it's running:
```bash
ps aux | grep 'dllama worker'
```

##### 2. Worker Failed to Start

**Symptom**: Worker process exits immediately

**Solution**: Check worker logs for errors

```bash
# Run worker with output to see errors
./dllama worker --port 9999 --nthreads 4 2>&1 | tee worker.log
```

Common errors:
- Port already in use → Choose different port
- Model file not found → Verify model path
- Memory allocation failure → Reduce nthreads or use smaller model

##### 3. Firewall Blocking Connection

**Symptom**: Port not accessible from root node

**Solution**: Allow port through firewall

Ubuntu/Debian (ufw):
```bash
sudo ufw allow 9999/tcp
sudo ufw reload
```

CentOS/RHEL (firewalld):
```bash
sudo firewall-cmd --add-port=9999/tcp --permanent
sudo firewall-cmd --reload
```

iptables:
```bash
sudo iptables -A INPUT -p tcp --dport 9999 -j ACCEPT
sudo iptables-save
```

##### 4. Cloud/VM Security Groups

**AWS**: Add inbound rule for port 9999 (TCP) in Security Group

**GCP**: Add firewall rule allowing tcp:9999

**Azure**: Add inbound security rule for port 9999

##### 5. Network Routing Issues

**Symptom**: Ping works but TCP connection fails

**Solution**: Check for network segmentation or routing rules

```bash
# Test TCP connectivity
nc -zv 100.78.3.114 9999

# Or use telnet
telnet 100.78.3.114 9999
```

##### 6. Worker Listening on Wrong Interface

**Symptom**: Worker runs but only listens on localhost

**Current behavior**: dllama worker listens on 0.0.0.0 (all interfaces) by default

**Verification**:
```bash
# On worker node
netstat -tlnp | grep 9999
# Should show: 0.0.0.0:9999

# Or with ss
ss -tlnp | grep 9999
```

### Connection Timeout (New in v0.4.4+)

As of the latest version, connection attempts will timeout after **30 seconds** instead of hanging indefinitely.

When a timeout occurs, you'll see:

```
Connection timeout after 30 seconds connecting to 100.78.3.114:9999

💡 Troubleshooting steps:
   1. Verify worker is running: ssh 100.78.3.114 'ps aux | grep dllama'
   2. Check port is listening: ssh 100.78.3.114 'netstat -tlnp | grep 9999'
   3. Test connectivity: ./diagnose_network.sh 100.78.3.114 9999
   4. Check firewall rules on both nodes
```

### Multi-Worker Setup

When connecting to multiple workers, each connection must succeed:

```bash
# Example: 2 workers
./dllama inference \
    --model model.m \
    --tokenizer tokenizer.t \
    --workers 192.168.1.10:9999 192.168.1.11:9999 \
    --prompt "test"
```

If **any** worker fails to connect, the entire initialization fails.

**Debugging**:
1. Test each worker individually
2. Check logs on each worker node
3. Verify all workers are using the same model version

### SSH Tunneling (Alternative for Firewall Issues)

If you cannot open firewall ports, use SSH tunneling:

```bash
# On root node, tunnel to worker
ssh -L 9999:localhost:9999 user@100.78.3.114

# Then connect to localhost
./dllama inference --workers localhost:9999 ...
```

For multiple workers, use different local ports:
```bash
# Terminal 1
ssh -L 10001:localhost:9999 user@worker1

# Terminal 2  
ssh -L 10002:localhost:9999 user@worker2

# Then connect
./dllama inference --workers localhost:10001 localhost:10002 ...
```

### Performance Issues After Connection

If connection succeeds but inference is slow:

1. **Network bandwidth**: Check with `iperf3`
   ```bash
   # On worker
   iperf3 -s
   
   # On root
   iperf3 -c 100.78.3.114
   ```

2. **Latency**: Check with `ping`
   ```bash
   ping -c 100 100.78.3.114 | tail -1
   ```

3. **CPU saturation**: Check with `top` on worker nodes

4. **Memory pressure**: Check with `free -h`

### Still Having Issues?

1. Run full diagnostic: `./diagnose_network.sh <worker_ip> <port>`
2. Check worker logs for errors
3. Verify network path with `traceroute`
4. Test with local worker first: `./dllama worker --port 9999`
5. Create minimal reproduction and file an issue

## Model Loading Issues

(To be added)

## Memory Issues

(To be added)

## Performance Issues

(To be added)
