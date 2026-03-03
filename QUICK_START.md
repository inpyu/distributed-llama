# Distributed-Llama Quick Start Guide

## ✅ 검증 완료: 2노드에서 정상 작동!

### 1. 로컬 테스트 (단일 머신, 2노드)

```bash
# Terminal 1: Worker 시작
./dllama worker --port 10001 --nthreads 4

# Terminal 2: Root에서 inference 실행
./dllama inference \
    --model dllama_model_original_q40.m \
    --tokenizer dllama_tokenizer_llama3_8B.t \
    --buffer-float-type q80 \
    --prompt "Hello world" \
    --steps 16 \
    --nthreads 4 \
    --collective star \
    --workers localhost:10001
```

### 2. 멀티 머신 설정 (예: 8노드)

**각 Worker 노드에서 (node 1-7):**
```bash
./dllama worker --port 9999 --nthreads 4
```

**Root 노드에서 (node 0):**
```bash
./dllama inference \
    --model dllama_model_original_q40.m \
    --tokenizer dllama_tokenizer_llama3_8B.t \
    --buffer-float-type q80 \
    --prompt "Hello world" \
    --steps 64 \
    --nthreads 4 \
    --collective auto \
    --workers \
        100.78.3.114:9999 \
        100.68.147.68:9999 \
        100.77.7.70:9999 \
        100.70.41.9:9999 \
        100.67.190.3:9999 \
        100.84.48.55:9999 \
        100.76.95.128:9999
```

### 3. 문제 해결

**연결 후 멈춤 현상:**
- 모든 worker 노드가 실제로 실행 중인지 확인
- 각 worker에서 "Listening on 0.0.0.0:9999..." 메시지 확인
- 방화벽 설정 확인
- 모델 로딩은 시간이 걸릴 수 있음 (몇 분 대기)

**Worker 상태 확인:**
```bash
# 각 worker 노드에서
ps aux | grep dllama
netstat -tuln | grep 9999
```

### 4. 자동화 스크립트

로컬 테스트용:
```bash
./test_single_worker.sh  # 2노드 테스트
./test_local_4nodes.sh   # 4노드 테스트 (localhost)
```
