# distributed-llama CPU-only (Raspberry Pi) Refactor Plan

목표: Raspberry Pi 기반 CPU 클러스터에서 `worker(=TP degree)`를 늘려도 **TTFT(Time To First Token)** 및 tail latency(p95/p99)가 악화되지 않도록, `distributed-llama`를 구조적으로 리팩토링하고 운영/측정까지 포함해 “재현 가능하게” 개선한다.

이 문서는 “GPU 없음(=CPU-only)”을 전제로 한다. 따라서 vLLM/TensorRT-LLM은 직접 마이그레이션 대상이 아니라, 그들이 쓰는 **원칙(배칭/스케줄링/통신 병목 완화)**만을 참고한다.

---

## 0. 현재 관측된 사실(레포 내부 근거)

### 0.1 TTFT가 worker 수 증가에 따라 악화되는 무릎(knee)이 존재

`distributed-llama/docs/SCALING_PERFORMANCE_REPORT.md`의 Raspberry Pi 클러스터 측정:

- 30B Q40: 2노드 TTFT ~16.0s, 4노드 TTFT ~16.1s, 8노드 TTFT ~37.8s
- 13B Q40: 2/4노드 TTFT ~7.7s, 8노드 TTFT ~20.9s

즉, **4노드가 무릎**으로 관측되며, 8노드에서 TTFT가 크게 증가한다.

참고(동일 문서의 부가 관측): 8노드에서

- 총 전송량이 2노드 대비 약 4배로 증가(선형 증가)
- 평균 네트워크 지연이 3~5ms대에서 10ms+로 증가
- `SYNC_NODE_SLICES` 평균 지연이 수 ms에서 수십 ms로 증가

이는 “노드가 늘수록 통신량/통신지연/동기화 지연이 TTFT를 지배”하는 패턴과 일치한다.

### 0.1.1 베이스라인(이 문서에서 고정할 기준값)

실험 및 리팩토링의 합격/실패 판정을 위해, 아래 베이스라인을 **그대로** 유지하며 비교한다.

- 워커 포트: `9999`
- `--nthreads`: 4(또는 장비 코어 수에 맞춰 명시적으로 고정)
- 프롬프트/스텝: 문서의 기본값(예: "Hello world", steps 128)

30B(Q40) 기준(요약):

| 노드 수 | Tokens/s | TTFT(s) | TPOT(ms) |
|--------|----------|---------|----------|
| 2      | ~0.39    | ~16.00  | ~2423    |
| 4      | ~0.44    | ~16.08  | ~2151    |
| 8      | ~0.18    | ~37.77  | ~5369    |

13B(Q40) 기준(요약, 비교 문서):

| 노드 수 | Tokens/s | TTFT(s) | TPOT(ms) |
|--------|----------|---------|----------|
| 2      | ~0.80    | ~7.67   | ~1196    |
| 4      | ~0.83    | ~7.71   | ~1145    |
| 8      | ~0.31    | ~20.88  | ~3084    |

근거:

- `distributed-llama/docs/SCALING_PERFORMANCE_REPORT.md`
- `distributed-llama/docs/SCALING_PERFORMANCE_REPORT_COMPARISON.md`

### 0.2 통신/동기화가 레이어 단위로 자주 발생하는 TP-by-dimension 구조

`distributed-llama/NODE_DISTRIBUTION_GUIDE.md`:

- Pipeline Parallelism(레이어 분할)이 아니라 Tensor Parallelism(텐서 차원 분할)
- KV cache도 차원 분할
- 레이어마다 `SYNC_NODE_SLICES` 동기화가 삽입됨

`distributed-llama/src/nn/nn-network.cpp`:

- `SYNC_NODE_SLICES`는 현재 **Star All-Reduce(루트 중심 gather+reduce+broadcast)**로 동작
- 구현상 `syncNodeSlices_starAllReduce`는 `threadIndex != 0`에서 early return(실질적으로 단일 스레드)

### 0.3 스레드 오버헤드가 노드 수 증가 시 지배적 병목으로 성장

`perf_scaling_comparison_2_4_8_nodes.md`:

- 8노드에서 Thread overhead가 ~50% 수준까지 증가(새 병목)

`distributed-llama/src/nn/nn-executor.cpp`:

- `NnExecutor::forward()`가 호출될 때마다 `pthread_create/join`을 반복(요청/토큰 스텝 수가 많을수록 누적)

### 0.4 운영 관례

- worker는 `./dllama worker --port 9999 --nthreads <n>`로 9999 포트에 상시 실행(`distributed-llama/README.md`, `distributed-llama/docs/SCALING_PERFORMANCE_REPORT.md`의 실행 예)
- 현장에서는 `kill_port_9999.sh`로 해당 포트 세션을 종료하는 운영이 있음(이 스크립트는 본 레포에 포함되어 있지 않을 수 있으므로 운영 자산으로 취급)

---

## 1. 문제 정의를 CPU-only에 맞게 재정렬

### 1.1 “worker 수 증가”가 의미하는 것

`distributed-llama`에서 worker 수 증가는 보통 “한 요청이 참여시키는 노드 수(=TP degree)” 증가를 의미한다.

- 장점: 노드 수가 늘면 메모리 분산으로 더 큰 모델을 올릴 수 있음
- 단점: 레이어마다 동기화가 있어, 노드 수가 늘수록 통신 latency/straggler가 누적되어 **TTFT/TPOT 악화 가능**

CPU-only(Raspberry Pi)에서는 이 문제가 더 심해진다.

- 네트워크: 보통 1GbE(또는 더 낮음), RDMA/IB 없음
- CPU: per-token compute가 느리므로 ‘통신을 compute로 숨기기’가 어렵고, tail이 쉽게 폭발
- 온도/전력: throttling으로 straggler가 발생하기 쉬움

### 1.2 목표를 2가지로 분리

1) **단일 요청의 TTFT/TPOT를 개선(혹은 악화 방지)**: TP 내부 최적화(스레드/통신/동기화)
2) **전체 처리량(throughput)을 스케일**: TP degree를 무작정 올리지 말고, “그룹(Replica)” 단위로 수평 확장

즉, “TP를 키워서 처리량 확장”이 아니라

- TP: 모델을 올릴 만큼만(그리고 무릎까지만)
- 그 이상: Replica(모델 그룹) 수를 늘려 throughput을 확장

으로 전략을 바꾼다.

추가로, CPU-only에서는 “단일 요청 TTFT 최적화”와 “전체 처리량 최적화”가 충돌하기 쉽다.

- 단일 요청 TTFT를 줄이려면: 큐잉을 줄이고(백프레셔), 통신 latency를 줄이고, 첫 flush를 빠르게
- 전체 처리량을 늘리려면: 적절한 배칭/동시성으로 CPU 활용도를 올리되, tail을 보호

따라서 본 계획은 **TTFT 보호(특히 p95/p99) 우선**을 기본 원칙으로 한다.

---

## 2. 아키텍처 방향성(권장 타겟)

### 2.1 기본 원칙

- **TP degree(노드 수)를 무릎까지 고정**: 실측상 4노드가 무릎(`distributed-llama/docs/SCALING_PERFORMANCE_REPORT.md`)
- **처리량은 Replica 그룹 수로 확장**: 4노드 그룹을 여러 개 만들고, 요청을 그룹에 분산
- **라우팅/큐잉/배칭은 별도 계층에서 제어**: TTFT를 지키려면 admission control(백프레셔)이 필수

추가 원칙(멀티 축 병렬화):

- TP(현재 방식)는 **무릎(<=4)까지**를 기본으로 하고, 그 이후의 확장은
  - DP(Replica)로 처리량을 늘리거나
  - 메모리 fit을 위해서만 PP(레이어 분할)를 도입
- CP(컨텍스트 병렬)는 “롱 프롬프트 prefill이 TTFT를 지배하는 경우”에만 연구 대상으로 둔다(대부분의 CPU+Ethernet 환경에서는 ROI가 낮을 수 있음).

### 2.2 권장 토폴로지(예)

클러스터 총 노드 수가 `2^k` 제약을 갖는 현재 특성(`distributed-llama/README.md`의 제한) 하에서:

- 그룹 크기 `G`: 4 (기본)
- Replica 수 `R = floor(TotalNodes / G)`

예) TotalNodes=8 이면

- (권장) 4노드 그룹 2개로 복제(throughput↑) 또는
- (필요 시) 8노드 단일 그룹로 모델 fit(메모리 목적)

중요: 그룹 크기(G)는 단순히 성능만이 아니라 **RAM fit**에 의해 결정된다.

- 30B 예시에서 4노드는 worker 메모리가 ~5.9GB, 8노드는 ~3.0GB 수준으로 관측됨(`distributed-llama/docs/SCALING_PERFORMANCE_REPORT.md`)
- 즉, “더 많은 노드”는 TTFT를 악화시킬 수 있지만, **노드당 RAM을 낮춰 모델을 올리는 목적**에는 필요할 수 있음

따라서 운영 정책은 아래 2단계로 결정한다.

1) 모델 fit을 위해 필요한 최소 노드 수를 먼저 결정(메모리 목적)
2) 그 노드 수가 TTFT 무릎을 넘는다면, TTFT를 지키는 대신 replica/라우팅/큐 정책으로 사용자 경험을 보호

### 2.3 엔드포인트 계층

이미 `dllama-api`가 OpenAI 유사 endpoint + 스트리밍(SSE/chunked)을 제공한다(`distributed-llama/src/dllama-api.cpp`).

단, 현재 구현은:

- 단일 프로세스/단일 inference context(사실상 단일 “그룹”만)
- accept loop가 직렬(연결마다 순차 처리)

따라서 “멀티 Replica 라우팅/동시성/백프레셔”를 위한 `gateway/api` 계층을 추가하거나, `dllama-api` 자체를 확장한다.

CPU-only에서 특히 중요한 포인트:

- `dllama-api`는 현재 `accept()` 이후 요청을 순차 처리하는 형태(`distributed-llama/src/dllama-api.cpp`의 `while (true)` 루프)
- 스트리밍 TTFT는 “첫 토큰 생성”뿐 아니라 “첫 chunk를 언제 flush하느냐”에도 영향을 받음
  - `writeStreamStartChunk`/`writeStreamChunk`/`fflush` 동작을 TTFT 측정에 포함해야 함

### 2.4 vLLM/TensorRT-LLM이 권장하는 PP/CP를 CPU-only로 번역하기

vLLM/TensorRT-LLM 문서들은 공통적으로 “TP만으로 스케일하면 통신/동기화가 병목이 되므로, PP/CP/DP 등 여러 축을 조합”하는 방향을 설명한다.

- vLLM: TP/PP/DP 전략 설명 및 네트워크가 느리면 TP가 비효율적일 수 있음을 명시
  - `vllm/docs/serving/parallelism_scaling.md`
  - `vllm/docs/configuration/optimization.md`(chunked prefill, TTFT/ITL 트레이드오프)
  - `vllm/docs/features/disagg_prefill.md`(prefill/decode 분리로 TTFT와 ITL을 별도 튜닝)
- TensorRT-LLM: 병렬화 축으로 TP/PP/DP/CP를 열거하고, CP는 “롱 컨텍스트”에 적합하다고 명시
  - `TensorRT-LLM/docs/source/features/parallel-strategy.md`

하지만 위 문서들은 기본적으로 GPU/NCCL/RDMA 같은 전제를 많이 깔고 있다. CPU-only Raspberry Pi 클러스터에서의 현실적인 번역은 다음이다.

- DP(Replica): 가장 현실적인 수평 확장 축(요청 단위 분산). TTFT를 안정화하기 가장 쉽다.
- PP(레이어 분할): “TP를 무릎(<=4)로 고정”하면서도 더 많은 노드를 활용해 **메모리 fit/throughput**을 얻기 위한 축.
  - 단, PP는 stage 간 activation 전송이 필수라서 **단일 요청 TTFT 자체가 좋아진다고 가정하면 안 됨**.
  - 대신 “TP=8로 무리해서 TTFT 폭발”하는 상황을 “PP=2 x TP=4” 같은 구성으로 완화할 수 있는지 A/B로 검증한다.
- CP(컨텍스트 병렬): CPU+Ethernet에서는 교차 노드 트래픽/조정 비용이 커서 대개 불리.
  - 도입 조건: 롱 프롬프트가 대부분이고 prefill이 TTFT의 지배 항인 워크로드에서만 연구.
  - 대안(현실적 CP-like): **prefill/decode 스케줄링 분리 + prefill chunking 정책**을 먼저 도입(Phase 0~3과 호환).

---

## 3. 원인 가설(측정으로 쪼개서 확정할 것)

### 3.1 8노드에서 TTFT가 급증하는 직접 원인 후보

`distributed-llama/docs/SCALING_PERFORMANCE_REPORT.md`에 따르면 8노드에서:

- 네트워크 총 전송량 증가(선형)
- `SYNC_NODE_SLICES` 평균 지연이 2노드 대비 크게 증가
- socket 0(루트 경로)이 병목으로 자주 관측

`distributed-llama/src/nn/nn-network.cpp`에서:

- `SYNC_NODE_SLICES`는 루트 중심 Star All-Reduce이며 단일 스레드 동작

즉, **루트가 모든 worker로부터 full-buffer를 순차적으로 받고 reduce한 뒤 다시 broadcast**하는 경로가 TTFT에 치명적인 후보.

### 3.2 노드 증가 시 thread overhead가 커지는 직접 원인 후보

`NnExecutor::forward()`가 매 호출마다 스레드를 만들고 join한다(`distributed-llama/src/nn/nn-executor.cpp`).

LLM decode는 “토큰당 forward 1회”이므로, 이 오버헤드는 토큰 수/동시요청에 비례하여 커진다.

---

## 4. 개발 로드맵(Phase별)

기간은 보통 6~10주를 가정한다(라즈베리파이 클러스터에서의 실험/프로파일링/원격 배포 시간을 포함).

### Phase 0 (Week 1): 측정 기준/재현성 확립

목표: “왜 느려졌는지”가 숫자로 갈라지도록 만든다.

작업:

- TTFT 분해 정의(필수)
  - `t_accept`(요청 수신)
  - `t_dispatch`(그룹/워커 선택 후 실행 시작)
  - `t_prefill_start` / `t_prefill_done`
  - `t_first_token`(첫 토큰 생성 시점)
  - `t_first_byte_sent`(클라이언트로 첫 chunk flush)
  - `t_done`
- 클러스터 측정 매트릭스 고정
  - 그룹 크기: 2 / 4 / 8 노드
  - 입력 길이: 256 / 1k / 4k / 8k tokens(가능 범위)
  - 출력 길이: 1(TTFT 중심) / 128(steady decode)
  - 동시성: 1 / 4 / 8 / 16
- 네트워크 모니터링/병목 리포트 자동 수집
  - `network->enablePerformanceMonitoring(true)` 경로 활용(`distributed-llama/src/app.cpp`)

완료 기준:

- 2/4/8 노드에서 “TTFT가 어디에서 증가했는지(스레드/동기화/네트워크/flush)”가 stage 단위로 분해되어 보인다.

추가 완료 기준(리그레션 방지):

- 동일 커밋/동일 설정에서 측정 결과가 3회 반복 시 p95 TTFT 편차가 허용 범위 내(예: 10% 미만)

### Phase 1 (Week 2~3): Thread 모델 리팩토링(가장 큰 ROI)

목표: 노드 수 증가 시 thread overhead 폭증을 억제한다.

핵심 작업(제안):

- `NnExecutor::forward()`의 “매 호출 pthread create/join” 제거
  - 고정 스레드 풀(persistent threads)로 전환
  - 스레드들은 condition variable 또는 futex 유사(가능 범위)로 대기
  - `currentStepIndex` 변화로 작업을 시작하고, 완료를 barrier로 모음

관련 코드:

- `distributed-llama/src/nn/nn-executor.cpp`

완료 기준:

- `perf` 기준으로 thread handler/관리 비중이 유의미하게 감소
- 8노드에서 p95/p99 TTFT 변동폭(지터)이 감소

권장 정량 목표(초기 가이드):

- `perf_scaling_comparison_2_4_8_nodes.md`에서 관측된 8노드 Thread overhead(약 50%)를
  - 1차: 35% 이하
  - 2차: 25% 이하
  수준으로 낮추는 것을 목표로 한다.

### Phase 2 (Week 3~5): SYNC_NODE_SLICES 통신 알고리즘/구현 최적화

목표: 8노드에서 `SYNC_NODE_SLICES` 평균 지연 및 tail을 줄여 TTFT 악화를 완화.

2.1 빠른 승리(낮은 위험)

- Star All-Reduce의 “루트 단일 스레드” 제거
  - 현재 `syncNodeSlices_starAllReduce()`는 `threadIndex != 0`을 return
  - 루트에서 worker readMany를 스레드로 분산하고, reduce도 chunk 단위로 병렬화

관련 코드:

- `distributed-llama/src/nn/nn-network.cpp` (`syncNodeSlices_starAllReduce`, `reduceSum`)

2.2 구조적 대안(중간 위험)

- ring 기반 all-reduce로 전환/옵션화
  - 이미 `syncNodeSlices_ringAllReduce()` 구현이 존재
  - `syncNodeSlices()`에서 선택 로직(예: 노드 수가 커지면 ring)을 도입
  - CLI 플래그로 `--collective {star,ring,auto}` 같은 선택지를 추가(자동은 경험칙 기반)

관련 코드:

- `distributed-llama/src/nn/nn-network.cpp` (`syncNodeSlices_ringAllReduce`, `syncNodeSlices`)

2.3 전송/소켓 레벨 최적화(측정 기반)

- `MAX_CHUNK_SIZE`(4096) 튜닝: 메시지 크기/RTT에 맞춰 최적화(무작정 키우지 말고 측정)
- `--net-turbo`(non-blocking) 사용 시 EAGAIN 스핀으로 CPU를 태우는지 확인 후 backoff/poll 도입

완료 기준:

- `distributed-llama/docs/SCALING_PERFORMANCE_REPORT.md`에서 관측된 8노드 TTFT 급증의 주된 증가분이 줄어듦
  - 최소 목표: 8노드 TTFT가 4노드 대비 “2배 이상” 튀는 현상을 완화

권장 정량 목표(30B 기준, 초기 가이드):

- 8노드 TTFT를 ~37.8s에서 ~25s 이하로(약 -30% 이상) 낮추기
- 또는(더 보수적으로) 8노드 TTFT의 증가분을 "4노드 대비 +50%" 이내로 제한

주의: 라즈베리파이 환경/네트워크 품질에 따라 목표치는 조정될 수 있으나, 반드시 “동기화/네트워크 stage의 감소”로 설명 가능해야 한다.

### Phase 3 (Week 5~7): “TP 무릎 고정 + Replica 스케일” 운영 구조 도입

목표: 처리량 확장은 TP degree 증가가 아니라, “그룹 복제 + 라우팅”으로 해결.

핵심 작업(2가지 옵션 중 택1 또는 병행):

옵션 A) 다중 `dllama-api` 인스턴스 + 외부 라우터

- 각 Replica 그룹(예: 4노드)마다 `dllama-api --workers <그룹>`를 별도 포트로 실행
- 앞단에 경량 라우터(프록시)를 두고 요청을 분산(라운드로빈/least-loaded)
- 장점: 기존 코어 변경 최소
- 단점: 포트/프로세스 관리 증가

옵션 B) `dllama-api` 내부에 Replica pool 내장

- `dllama-api`가 여러 개의 `RootLlmInference` context를 보유(그룹별)
- 요청마다 그룹을 선택하고 그 context로 실행
- admission control(큐 제한, 동시 실행 제한)을 내장
- 장점: 단일 엔드포인트
- 단점: 구현 난이도 상승(동시성/자원 관리)

관련 코드:

- `distributed-llama/src/dllama-api.cpp`

운영 제약 반영:

- worker는 계속 `:9999` 유지(현장 운영과 호환)
- 그룹 구성은 현재 제약(2^n 노드)에 맞추어 시작

완료 기준:

- “노드 수를 늘려도 TTFT가 악화되지 않음”을 **처리량 스케일**로 달성
  - 예: 8노드를 8노드 단일 TP로 쓰는 대신, 4노드 그룹 2개로 운영했을 때
    - 단일 요청 TTFT는 4노드 수준 유지
    - 전체 동시 처리량은 증가

권장 정량 목표(예시):

- 8노드를 (8노드 TP 1개) 대신 (4노드 TP 2개 replica)로 운영할 때
  - 단일 요청 TTFT p95는 4노드 TP와 유사
  - 동일 TTFT 목표를 유지하면서 처리 가능한 동시 요청 수(throughput)가 증가

### Phase 4 (Week 7~10): PP/CP 도입 스파이크(근거 기반, PP 우선)

현실 인식(중요): 현재 구조는 TP-by-dimension이며 레이어마다 동기화가 박혀 있다(`distributed-llama/NODE_DISTRIBUTION_GUIDE.md`). PP/CP를 “제대로” 넣는 것은 **거의 새 엔진 수준**의 변경이다.

그럼에도 vLLM/TensorRT-LLM이 PP/CP를 권장하는 이유(=TP-only 한계)를 반영해, Phase 4에서는 “가능한 최소 리스크”로 PP/CP를 **스파이크**하고, 수치로 go/no-go를 결정한다.

따라서 Phase 4는 아래 조건을 만족할 때만 진행한다.

- Phase 1~3을 했는데도 TTFT 목표를 달성하지 못함
- 또는 특정 모델이 메모리/성능 요구로 인해 PP 없이는 운영이 불가능

#### Gate 0: TP 무릎 원인 확정

- TP=1/2/4/8에서 `compute vs SYNC_NODE_SLICES` 비중과 TTFT p95를 비교
- 8에서 sync가 TTFT를 지배(또는 superlinear)하는 것이 확인되면 PP 스파이크 진행

#### Gate 1: 스레드 모델/분산 변수 제거

- Phase 1(스레드 모델 리팩토링)으로 forward-time 분산을 낮춘 뒤에만 PP/CP를 평가

#### PP 스파이크(권장 1순위)

Oracle 권장(요약): CPU-only Ethernet에서는 PP가 “가치가 있을 수 있는 유일한 추가 축”이지만, 단일 요청 TTFT를 마법처럼 줄이는 것이 아니라 **TP를 무릎(<=4)로 고정하면서도 더 많은 노드를 활용**하기 위한 수단으로 봐야 한다.

권장 초기 구성:

- 8노드라면 `PP=2 x TP=4`부터 시작(=stage boundary 최소화)
- `PP=4 x TP=2`는 stage boundary가 늘어 TTFT에 불리할 확률이 높으므로 후순위

가장 단순한 PP 설계(least-risk):

- 레이어를 연속 구간으로 2등분(앞 절반/뒤 절반)
- Stage 0: embedding + 초기 N layers
- Stage 1: 나머지 layers + final norm/lm-head
- stage 간에는 hidden state(activation)만 전달

스케줄링 원칙:

- decode: TTFT 단독 개선이 목적이 아니라, 동시 요청 배칭으로 throughput을 올리면서 TTFT 예산 내 유지
- prefill: 긴 프롬프트에서만 token-chunk microbatch(예: 16~64 tokens)를 시도하되, chunk가 너무 작아지면 TTFT가 증가할 수 있으므로 측정 기반으로만 적용

측정/합격 기준(예시):

- 비교군: `TP=8` vs `PP=2 x TP=4` (동일 모델/동일 프롬프트/동일 동시성)
- 보고 지표:
  - TTFT p50/p95
  - steady tokens/s
  - `SYNC_NODE_SLICES` 시간 vs stage 전송 시간
  - bytes/token, CPU idle(wait) 비중
- stop condition:
  - `PP=2 x TP=4`가 throughput 이득이 없고, TTFT p95가 허용 예산(예: 1.3x) 이상 악화되면 즉시 중단

#### CP 스파이크(조건부, 2순위)

TensorRT-LLM은 CP를 “롱 컨텍스트”에 적합하다고 명시한다(`TensorRT-LLM/docs/source/features/parallel-strategy.md`).

CPU-only에서 CP는 아래 조건을 만족할 때만 진행:

- 워크로드의 대부분이 장문 프롬프트이고, TTFT의 지배 항이 prefill
- 그리고 PP/DP만으로는 TTFT/throughput 목표를 달성하지 못함

그 외에는 CP 대신 아래 CP-like를 우선:

- prefill/decode 스케줄링 분리(동일 엔진 내에서 우선순위로 간섭 최소화)
- prefill chunking 정책(=vLLM의 chunked prefill 아이디어를 “CPU 스케줄러”로 번역)
  - vLLM은 `max_num_batched_tokens` 튜닝으로 TTFT/ITL 트레이드오프를 설명(`vllm/docs/configuration/optimization.md`).

연구 항목(원형):

- PP(레이어 분할) 설계: 레이어 stage 간 activation send/recv 프로토콜, 파이프라인 버블/스케줄링
- CP(컨텍스트 축 분할) 설계: prefill에서만 sequence shard 적용 가능성 검토

완료 기준:

- “CPU-only 이더넷 환경에서도” 통신량/동기화 빈도 감소가 실제로 이득인지 수치로 입증
- 입증 실패 시: 기능 구현을 중단하고 Phase 2/3의 최적화에 집중

---

## 5. 실험 설계(반드시 지켜야 할 규칙)

### 5.1 워크로드 2개로 분리

- Prefill-only: 출력 1토큰(또는 매우 짧게)로 TTFT만 본다.
- Decode-steady: 출력 길게(128/512)로 TPOT/throughput을 본다.

근거: `distributed-llama-scaling-analysis.md`가 동일한 분리 측정 필요성을 이미 강조.

### 5.2 비교군을 섞지 않기

- “TP degree 확장(한 요청에 참여하는 노드 수)” 실험과
- “Replica 확장(동시 요청을 분산 처리)” 실험을 분리한다.

섞으면 TTFT 악화의 원인이 (통신/스레드)인지 (큐잉/라우팅)인지 구분이 안 된다.

### 5.3 핵심 그래프 3개(보고서에 고정)

1) `TTFT p95 vs TP degree`
2) `TTFT p95 vs Replica count (고정 TP degree)`
3) `tokens/s vs concurrency` (각 구성별)

---

## 6. 운영/배포(라즈베리파이 현실 반영)

### 6.1 노드 표준화 체크리스트

- CPU governor 고정(성능 모드) + thermal throttle 모니터링
- 동일한 바이너리/모델 파일 버전(릴리즈/커밋 고정)
- 네트워크: 유선, 스위치 품질 확인, MTU 통일
- SD카드 I/O 병목 제거(가능하면 SSD/USB3)

추가(네트워크):

- 가능하면 동일 스위치/동일 케이블 품질로 통일
- 와이파이 사용 금지(특히 TTFT tail 악화)

추가(프로세스/리소스):

- OOM 발생 시 재시작 폭풍을 막기 위해, 그룹 단위 admission control(동시 요청 제한)을 먼저 적용

### 6.2 프로세스 관리

현재 관례:

- worker: `./dllama worker --port 9999 --nthreads <n>`
- stop: `kill_port_9999.sh`(운영 자산)

계획:

- 최소: PID 파일 + healthcheck + 롤링 재시작 스크립트
- 권장: systemd unit(자동 재시작, 로그 집계)

### 6.3 장애 모드(반드시 대비)

- 좀비 프로세스가 9999 점유 → 새 worker가 못 뜸
- 일부 노드 throttling → sync straggler로 TTFT tail 폭발
- 네트워크 순간 혼잡 → `SYNC_NODE_SLICES` tail 상승

대응:

- 그룹에서 느린 노드 자동 제외(circuit breaker)
- admission control로 과부하 시 빠른 실패(큐 적체 방지)

---

## 7. 산출물(Deliverables)

- `TTFT/TPOT/throughput` 측정 스크립트 + 결과 저장 포맷(JSON/CSV)
- perf/flamegraph 기반 병목 리포트 템플릿
- Phase 1~3 변경사항에 대한 리그레션 실행 방법(테스트 바이너리 + 재현 커맨드)
- 운영 런북: 노드 추가/제거, 롤링 재시작, 장애 대응

---

## 8. 즉시 실행 우선순위(요약)

1) Phase 0: TTFT 분해 측정부터 고정
2) Phase 1: `NnExecutor::forward()` 스레드 모델을 persistent로 바꿔 thread overhead를 먼저 잡기
3) Phase 2: `SYNC_NODE_SLICES`의 루트 단일 스레드/루트 병목을 제거(또는 ring 옵션 도입)
4) Phase 3: “4노드 그룹 고정 + Replica 라우팅”으로 throughput을 스케일

---

## Appendix A. 관련 파일/터치포인트(작업용 인덱스)

- 실행/운영
  - `distributed-llama/README.md`
  - `distributed-llama/docs/HOW_TO_RUN_RASPBERRYPI.md`
  - `distributed-llama/docs/SCALING_PERFORMANCE_REPORT.md`
- API/스트리밍
  - `distributed-llama/src/dllama-api.cpp`
- 스레드/실행기
  - `distributed-llama/src/nn/nn-executor.cpp`
  - `distributed-llama/src/nn/nn-executor.hpp`
- 통신/동기화
  - `distributed-llama/src/nn/nn-network.cpp`
  - `distributed-llama/src/nn/nn-network.hpp`
- 분산 정책 설명
  - `distributed-llama/NODE_DISTRIBUTION_GUIDE.md`
  - `distributed-llama/docs/COMMUNICATION_BOTTLENECK_PARALLEL_POLICY.md`
