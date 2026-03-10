# worker 증가 시 성능이 개선되지 않은 원인 분석 (plan.md 기반)

Last updated: 2026-03-03

이 문서는 `distributed-llama/plan.md`의 의도(Phase 3~4 방향)와 `distributed-llama/중간개선.md`의 측정 결과를 대조해, "worker를 늘렸는데 왜 성능이 안 좋아지는가"를 코드 근거로 설명하고, 앞으로의 개선 방향/실험 설계를 정리한다.

## 1) 관측 요약 (중간개선.md)

`distributed-llama/중간개선.md`의 측정에서 공통적으로:

- Workers(=노드/TP degree) 증가 시 `tokens/s`가 감소하고, `TTFT`/`Decode(ms)`가 증가한다.
- 개선 후(`--collective auto`)라고 표기돼 있어도 2->4->8에서 strong scaling이 되지 않는다.

즉, 단일 요청을 더 많은 worker로 쪼개 처리하는 방식(=strong scaling)이 현재 구조/환경에서 성립하지 않는다.

## 2) plan.md가 기대하는 “스케일링”과 현재 실험의 불일치

`distributed-llama/plan.md`는 이미 다음을 전제로 한다:

- 4노드 이후(특히 8노드)에서 TTFT가 동기화/네트워크 tail에 의해 지배된다.
- throughput 확장은 TP degree를 무작정 올리는 것이 아니라, **TP 무릎(knee)을 고정하고 replica를 늘리는 방식**(Phase 3)을 우선으로 둔다.

따라서 `중간개선.md`에서의 "Workers↑ => 단일 요청 tokens/s↑" 기대는, `plan.md`의 Phase 3 방향(Replica 스케일)과 애초에 목표가 다르다.

## 3) 왜 worker 증가가 성능 개선으로 이어지지 않는가 (코드 근거)

### 3.1 workers는 “동시 요청 처리량”이 아니라 “단일 요청의 TP 분할 정도”에 가깝다

- 분산 방식은 PP(레이어 분할)가 아니라 TP(텐서 차원 분할)이다.
  - 근거: `distributed-llama/NODE_DISTRIBUTION_GUIDE.md`
- 단일 요청의 decode는 `batchSize=1`로 1토큰씩 forward를 반복한다.
  - 근거: `distributed-llama/src/dllama.cpp` (Prediction 루프에서 `setBatchSize(1)`)

즉 worker를 늘려도 “동시에 여러 요청을 처리”하는 구조가 아니면, 전체 처리량(throughput)은 늘지 않는다.

### 3.2 per-token/per-layer 동기화가 구조적으로 강제된다

TP 구조에서는 각 레이어에서 부분 결과를 합쳐 다음 연산으로 넘어가야 하므로 동기화가 반복된다.

- 레이어별 동기화 삽입(대표):
  - attention 블록 이후 `SYNC_NODE_SLICES`
  - feed-forward 블록 이후 `SYNC_NODE_SLICES`
  - logits 이후 `SYNC_NODE_SLICES`
  - 근거: `distributed-llama/src/llm.cpp`

decode(1 token/step)에서는 한 step의 compute가 작아, 통신 latency를 compute로 숨길 여지가 작아지고, 동기화 비용이 누적돼 역스케일링이 발생하기 쉽다.

### 3.3 `--collective auto`가 실제로 auto가 아니다 (항상 STAR로 강제)

`collectiveType=COLLECTIVE_AUTO`는 현재 `COLLECTIVE_STAR`로 고정된다.

- 근거: `distributed-llama/src/nn/nn-network.cpp`에서
  - `if (effective == COLLECTIVE_AUTO) effective = COLLECTIVE_STAR;`

따라서 `중간개선.md`의 "--collective auto" 실험은 코드 기준으로 "--collective star"와 동일하게 해석해야 한다.

### 3.4 STAR all-reduce 구현이 “full buffer 전송”이라 worker 증가 시 통신량/병목이 급증한다

현재 `syncNodeSlices_starAllReduce`는 root가 worker들로부터 **각 worker의 `nBytes` 전체 버퍼**를 모아(`readMany`), root에서 합산(`reduceSum`) 후, 다시 worker들에게 **`nBytes` 전체를 broadcast**한다.

- 근거: `distributed-llama/src/nn/nn-network.cpp`의 `syncNodeSlices_starAllReduce(...)`
  - root: `totalGatherSize = nBytes * nWorkers`
  - worker->root: `sendIo.size = nBytes`
  - root->worker: `writeIos[w].size = nBytes`

이 설계는 "all-gather를 all-reduce(sum)로 에뮬레이션"(각 노드가 자기 slice 외 영역은 0으로 채우고 sum으로 전체를 재구성)하는 방식과 맞물려,

- worker 수가 늘수록 per-sync 통신량이 선형 이상으로 커지고
- root의 소켓/CPU가 병목이 되며
- tail latency가 급격히 악화된다.

### 3.5 dllama-api는 accept 루프가 직렬이며, 단일 inference context를 공유한다

Phase 3에서 "replica 스케일"을 권장하는 이유를 코드 레벨에서도 확인할 수 있다.

- `distributed-llama/src/dllama-api.cpp`의 `server()`는 `while (true) { accept -> read -> resolve }` 형태의 직렬 처리다.

따라서 단일 `dllama-api` 인스턴스에 worker를 더 붙여도 동시 요청 처리량이 늘지 않는다. 처리량을 늘리려면 replica(프로세스/포트) 수평 확장 + 라우팅이 필요하다.

## 4) 이번 “개선이 안 보이는” 직접 원인 (정리)

`중간개선.md`에서 worker 증가 효과가 나타나지 않은 이유는 크게 2축이다.

1) **목표/실험 불일치**: workers 증가는 단일 요청 TP strong scaling에 가깝고, plan은 knee 이후 replica 스케일(throughput)을 목표로 한다.
2) **구현상 병목**:
   - decode는 per-token이고, 레이어마다 동기화가 들어간다.
   - `--collective auto`는 사실상 star 고정이다.
   - star all-reduce가 full buffer 전송이라 worker 증가 시 통신/루트 병목이 급증한다.

## 5) 앞으로 어떻게 개선할 것인가 (우선순위)

### 5.1 (운영/Phase 3) knee 고정 + replica 스케일로 throughput을 올린다

`plan.md`의 권장 운영을 그대로 따라가는 것이 가장 빠른 ROI다.

- 그룹 크기 `G=4` 고정
- `N=8`이면 `4노드 그룹 x 2개 replica`
- 라우터/게이트웨이로 요청 분산 + replica별 in-flight 제한(기본 1) + 큐 제한/429

근거: `distributed-llama/plan.md`의 Phase 3 (2장)

### 5.2 (정확한 원인 확정) star vs ring A/B를 “명시적으로” 재측정한다

현재 auto가 star로 고정이므로, 최소한 아래 A/B는 꼭 분리해 측정해야 한다.

- `--collective star`
- `--collective ring`

그리고 prefill/decode를 분리해 기록해야 한다.

- prefill: chunking 설정(`--prefill-chunk-size`, `--prefill-chunk-threshold`)을 함께 기록
- decode: 동일 output token 수에서 `Pred`/`Sync` 비중(로그) 비교

### 5.3 (구현/Phase 2.5 보강) COLLECTIVE_AUTO의 의미를 복원한다

`COLLECTIVE_AUTO`가 현재 star 고정인 상태는 문서/사용자 기대와 어긋난다.

추천 접근:

- AUTO는 **명시적 휴리스틱**으로 선택한다(예: nNodes, payload size, onlyFromWorkerToRoot 여부).
- AUTO 선택 결과를 실행 시작 시 1회 로그로 출력한다(실험 해석 오류 방지).
- 기본값을 바꾸기 전에 `plan.md`가 말하는 8노드 A/B(정확성/TTFT/안정성)를 먼저 통과시킨다.

### 5.4 (구조적) 통신량을 줄이는 알고리즘으로 전환한다

현재 star all-reduce는 full buffer 전송이라 통신량이 크다. 근본적으로는 다음 계열이 필요하다.

- reduce-scatter + all-gather (ring/tree 계열)
- slice 기반 전송(필요한 chunk만 전송) + 메시지 코얼레싱

이미 코드에는 ring all-reduce 구현이 존재한다.

- 근거: `distributed-llama/src/nn/nn-network.cpp`의 `syncNodeSlices_ringAllReduce(...)`

### 5.5 (Phase 4) PP=2 x TP=4로 “단일 요청에서 더 많은 노드 활용”을 연다

단일 요청에서 8노드를 쓰는 목적이 메모리 fit이라면, TP=8 강행이 아니라 PP 도입이 맞다.

- 근거: `distributed-llama/plan.md`의 Phase 4

## 6) 추천 실험 템플릿 (다음 측정에서 문서에 남길 것)

- 환경: 모델/quant, `--buffer-float-type`, nthreads, net-turbo, 네트워크 상태
- 워크로드 2종 분리:
  - prefill-only (입력 길이 L, 출력 1토큰)
  - decode-steady (prefill 후 D 토큰 생성)
- 지표:
  - TTFT, tokens/s, p50/p95/p99
  - `Pred` vs `Sync` 시간 비율(로그)
  - sent/recv bytes (로그)
- 실험 매트릭스:
  - TP: 2/4/8
  - collective: star/ring
  - replica: 1/2 (G=4 고정)
