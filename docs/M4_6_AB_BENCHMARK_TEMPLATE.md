# Phase 4 M4.6 A/B 벤치마크 + 튜닝 리포트 템플릿

이 문서는 `plan.md`의 M4.6 요구사항을 검증하기 위한 표준 리포트 템플릿입니다.

---

## 1) 실험 개요

- 실험 목적:
  - [ ] 대조군 `TP=8` 대비 실험군 `PP=2 x TP=4`의 TTFT 개선/동등성 검증
  - [ ] 안정성(1000회 연속 요청) 검증
  - [ ] PP=1 회귀 무결성 검증
- 실험 일시: `<YYYY-MM-DD HH:mm TZ>`
- 실행자/브랜치/커밋:
  - Branch: `<branch>`
  - Commit: `<sha>`

## 2) 테스트 환경

- 모델: `<model path>`
- 토크나이저: `<tokenizer path>`
- 버퍼 타입: `<q80/f16/...>`
- 프롬프트 세트: `<single|multi prompt set name>`
- 하드웨어:
  - Root: `<CPU/GPU/RAM>`
  - Worker: `<count, CPU/GPU/RAM each>`
- 네트워크:
  - 대역폭: `<Gbps>`
  - RTT(P50/P95): `<ms>`
- 공통 실행 옵션:
  - `--nthreads <n>`
  - `--n-batches <n>` (해당 시)
  - `--collective <auto|star|ring>`
  - `--buffer-float-type <...>`

## 3) 실험 매트릭스

| ID | Group | Topology | Workers | pp-size | tp-size | PromptLen Bucket | Steps | Runs |
|----|-------|----------|---------|---------|---------|------------------|-------|------|
| C1 | Control | TP=8 | 8 | 1 | 8 | Short | `<n>` | `<n>` |
| C2 | Control | TP=8 | 8 | 1 | 8 | Long | `<n>` | `<n>` |
| E1 | Experiment | PP=2 x TP=4 | 8 | 2 | 4 | Short | `<n>` | `<n>` |
| E2 | Experiment | PP=2 x TP=4 | 8 | 2 | 4 | Long | `<n>` | `<n>` |

권장 버킷 예시:
- Short prompt: `<64~256 tokens>`
- Long prompt: `<1024+ tokens>`

## 4) 실행 커맨드 (복붙 템플릿)

### 4.1 Control (TP=8)

```bash
./dllama inference \
  --model <model> \
  --tokenizer <tokenizer> \
  --prompt "<prompt>" \
  --steps <steps> \
  --workers <w1:port> ... <w8:port> \
  --pp-size 1 \
  --collective <auto|star|ring> \
  --nthreads <n> \
  --buffer-float-type <type>
```

### 4.2 Experiment (PP=2 x TP=4)

```bash
./dllama inference \
  --model <model> \
  --tokenizer <tokenizer> \
  --prompt "<prompt>" \
  --steps <steps> \
  --workers <w1:port> ... <w8:port> \
  --pp-size 2 \
  --prefill-chunk-threshold <n> \
  --prefill-chunk-size <0 or n> \
  --collective <auto|star|ring> \
  --nthreads <n> \
  --buffer-float-type <type>
```

### 4.3 PP=1 회귀 확인

```bash
make dllama nn-cpu-test nn-cpu-ops-test tokenizer-test nn-topology-test nn-pipeline-test
./nn-cpu-test
./nn-cpu-ops-test
./tokenizer-test
./nn-topology-test
./nn-pipeline-test
```

## 5) 측정 항목 정의

- **TTFT(ms)**: `Timing -> ttftMs`
- **Prefill(ms)**: `Timing -> prefillMs`
- **Decode(ms)**: `Timing -> decodeMs`
- **Total(ms)**: `Timing -> totalMs`
- **Pred tokens/s**: `Prediction -> tokens/s`
- **Network stats**: `Network Performance Report` / `Bottleneck Analysis`
- **오류율**: 비정상 종료, timeout, deadlock, socket error 비율

## 6) Raw 결과 기록

### 6.1 실행별 결과 (원본)

| RunID | Group | PromptID | TTFT(ms) | Prefill(ms) | Decode(ms) | Total(ms) | Pred tokens/s | ExitCode | Error |
|------|-------|----------|----------|-------------|------------|-----------|---------------|----------|-------|
| `<id>` | `<C/E>` | `<p>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<0/1>` | `<none/msg>` |

### 6.2 집계 결과

| Group | Prompt Bucket | TTFT p50 | TTFT p95 | Pred tokens/s avg | Error Rate |
|-------|----------------|----------|----------|-------------------|------------|
| Control (TP=8) | Short | `<...>` | `<...>` | `<...>` | `<...>` |
| Experiment (PP=2xTP=4) | Short | `<...>` | `<...>` | `<...>` | `<...>` |
| Control (TP=8) | Long | `<...>` | `<...>` | `<...>` | `<...>` |
| Experiment (PP=2xTP=4) | Long | `<...>` | `<...>` | `<...>` | `<...>` |

## 7) 안정성 테스트 (1000회)

- 테스트 시나리오: `<same prompt | prompt set>`
- 반복 횟수: `1000`
- 실패 기준: 비정상 종료/timeout/deadlock/socket error

| Group | Attempts | Success | Fail | Error Rate | 주요 오류 유형 |
|-------|----------|---------|------|------------|-----------------|
| Control (TP=8) | 1000 | `<...>` | `<...>` | `<...>` | `<...>` |
| Experiment (PP=2xTP=4) | 1000 | `<...>` | `<...>` | `<...>` | `<...>` |

## 8) 합격 기준 체크 (plan.md 기준)

- [ ] **TTFT p95**: `PP=2xTP=4`가 `TP=8` 대비 개선 또는 최소 동등
- [ ] **안정성**: 1000회 연속 요청에서 오류 없음(또는 허용 오차 기준 충족)
- [ ] **회귀**: PP=1 모드가 기존과 동일 (빌드 + 회귀 테스트 통과)

결론:
- 판정: `<PASS | FAIL | CONDITIONAL PASS>`
- 핵심 근거 3줄:
  1. `<...>`
  2. `<...>`
  3. `<...>`

## 9) 튜닝 실험 로그

| Trial | 변경점 | 기대 효과 | 실제 효과 | 유지 여부 |
|-------|--------|-----------|-----------|-----------|
| T1 | `<prefill-chunk-threshold>` | `<...>` | `<...>` | `<Y/N>` |
| T2 | `<prefill-chunk-size>` | `<...>` | `<...>` | `<Y/N>` |
| T3 | `<collective mode>` | `<...>` | `<...>` | `<Y/N>` |

## 10) 리스크/이슈/후속 액션

- 리스크:
  - `<bandwidth bottleneck / stage imbalance / deadlock risk ...>`
- 발견 이슈:
  - `<issue id/summary>`
- 후속 액션:
  1. `<...>`
  2. `<...>`

---

## 부록 A) 로그 캡처 규칙

- 각 실행별로 stdout/stderr를 별도 파일에 저장
- 실패 케이스는 root/worker 양쪽 로그를 함께 보관
- 최소 보관 항목:
  - 실행 커맨드
  - 종료 코드
  - Timing 블록
  - Network Performance Report 블록

## 부록 B) 파일명 컨벤션(권장)

- `reports/m46/<date>/<group>_<promptId>_<runId>.log`
- `reports/m46/<date>/summary.csv`
- 최종 문서: `docs/M4_6_BENCHMARK_REPORT_<date>.md`
