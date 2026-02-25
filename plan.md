# distributed-llama CPU-only Roadmap (Phase 3~4 Execution Plan)

Last updated: 2026-02-25

목표: Raspberry Pi 기반 CPU 클러스터에서 (1) 단일 요청의 TTFT/TPOT 악화를 방지하고(가능하면 개선), (2) 전체 처리량(throughput)을 확장하되 tail latency(p95/p99)를 보호한다.

전제: CPU-only 환경을 유지한다. vLLM/TensorRT-LLM은 마이그레이션 대상이 아니라, 원칙(병렬화 축 조합/스케줄링/통신 병목 완화)만 참고한다.

---

## 0. Phase 2.5까지 진행 내용 검토(현황 + 리스크)

### 0.1 관측된 성능 특성(무릎)

근거:

- `distributed-llama/docs/SCALING_PERFORMANCE_REPORT.md`
- `distributed-llama/docs/SCALING_PERFORMANCE_REPORT_COMPARISON.md`

요약(대표값):

- 30B Q40: 2노드 TTFT ~16.0s, 4노드 TTFT ~16.1s, 8노드 TTFT ~37.8s
- 13B Q40: 2/4노드 TTFT ~7.7s, 8노드 TTFT ~20.9s

결론: 4노드 이후(특히 8노드)부터 동기화/네트워크 tail이 TTFT를 지배한다.

### 0.2 Phase 0~2.5 “완료”의 의미(검증 기준)

이 문서는 “2.5까지 개발 진행”을 전제로 Phase 3~4를 구체화하되, 아래 항목은 반드시 코드/측정으로 확인한다.

- Phase 0: 재현 가능한 측정 기준(특히 스트리밍 포함 TTFT 분해)과 반복 측정 변동폭 관리
- Phase 1: `NnExecutor::forward()` 호출당 스레드 생성/조인 제거(지속 스레드)로 thread overhead 감소
- Phase 2: `SYNC_NODE_SLICES` 병목 완화(알고리즘 옵션 확보 및 8노드에서 tail 개선)
- Phase 2.5: collective 선택 로직/안정성(특히 8노드) 정리

### 0.3 Phase 2.5에서 특히 재확인할 포인트(코드/문서 불일치 가능)

확인 대상(코드):

- `--collective {auto,star,ring}` 플래그 존재: `distributed-llama/src/app.cpp`
- ring all-reduce 존재: `distributed-llama/src/nn/nn-network.cpp` (`syncNodeSlices_ringAllReduce`)
- `COLLECTIVE_AUTO`가 실제로 노드 수에 따라 ring을 기본 선택하는지 확인 필요
  - 현재 코드에서는 `COLLECTIVE_AUTO`가 실질적으로 `COLLECTIVE_STAR`로 고정될 수 있음(`syncNodeSlices()` effective 결정 로직)

Phase 3~4 진입 전 최소 액션:

- 8노드에서 `--collective star` vs `--collective ring` A/B 결과(정확성/TTFT/안정성)로 “2.5 완료”를 확정

---

## 1. Phase 3~4 공통 원칙

### 1.1 목표를 2가지로 분리

1) 단일 요청 TTFT/TPOT 보호(가능하면 개선): TP 내부 최적화(스레드/통신/동기화)
2) 전체 처리량 확장: TP degree를 무작정 올리는 대신 Replica(그룹) 수평 확장 + 라우팅/큐/백프레셔

### 1.2 운영 토폴로지 기본값

- 기본 그룹 크기 `G=4`(무릎까지 고정)
- 총 노드 `N=8`이면 권장 운영은 `4노드 그룹 x 2개 replica`
- 더 많은 노드를 단일 요청에 쓰는 목적은 “메모리 fit”일 때만(그 경우 Phase 4: PP로 접근)

### 1.3 빌드/회귀 테스트(레포 CI 기준)

CI와 동일한 최소 검증:

```sh
make dllama dllama-api nn-cpu-test nn-cpu-ops-test tokenizer-test \
  && ./nn-cpu-test \
  && ./nn-cpu-ops-test \
  && ./tokenizer-test
```

근거:

- `distributed-llama/.github/workflows/main.yml`
- `distributed-llama/AGENTS.md`

---

## 2. Phase 3 (Week 1~2): "TP 무릎 고정 + Replica 스케일" 운영 구조 도입

Phase 3의 목표는 단일 요청에 노드를 더 붙이는 것이 아니라, **여러 요청을 동시에 처리**해서 throughput을 올리면서 단일 요청 TTFT(p95/p99)를 보호하는 것이다.

### 2.1 권장 접근(옵션 A부터)

옵션 A(권장): 다중 `dllama-api` 인스턴스 + 외부 라우터(게이트웨이)

- 장점: 코어 변경 최소, replica 단위 격리로 안정적
- 전제: `dllama-api`는 단일 inference context를 사용하며 accept 루프가 직렬임(`distributed-llama/src/dllama-api.cpp`)

옵션 B(후속/선택): `dllama-api` 내부 Replica pool 내장

- 장점: 단일 엔드포인트
- 단점: 동시성/자원관리/상태충돌 위험이 커서 Phase 3의 “첫 구현”으로는 비권장

### 2.2 Phase 3 산출물(Deliverables)

- D3.1: Replica 구성 스펙 + 예시 토폴로지
  - 예: `replicas.json` 또는 단순 텍스트("replica name -> base_url")
- D3.2: 게이트웨이(라우터)
  - 기능: healthcheck, round-robin/least-loaded, replica별 in-flight 제한(기본 1), 큐 제한, 타임아웃
- D3.3: 운영 런북
  - 기동/재시작/롤백, 포트 점유/느린 노드/네트워크 혼잡 대응
- D3.4: Phase 3 측정 리포트 템플릿
  - 그래프: `TTFT p95 vs replica count(고정 G=4)`, `tokens/s vs concurrency`

실행 예시(개념):

```sh
# Replica A (root node)
./dllama-api --port 9001 --model <...> --tokenizer <...> --buffer-float-type q80 --nthreads 4 --max-seq-len 4096 \
  --workers 10.0.0.2:9999 10.0.0.3:9999 10.0.0.4:9999

# Replica B (root node)
./dllama-api --port 9002 --model <...> --tokenizer <...> --buffer-float-type q80 --nthreads 4 --max-seq-len 4096 \
  --workers 10.0.0.6:9999 10.0.0.7:9999 10.0.0.8:9999

# Gateway (새로 구현)
./dllama-gateway --listen 9999 --backends http://10.0.0.1:9001 http://10.0.0.5:9002
```

### 2.3 작업 항목(마일스톤)

M3.1 (0.5~1d): "replica 운영" 실행 단위 정의

- 각 replica는 (root 1대 + worker 3대) 고정 그룹
- replica root마다 `dllama-api`를 다른 포트로 실행
- replica는 "동시에 1개 요청"만 처리하는 정책을 기본값으로 둔다(단일 context 보호)

M3.2 (1~2d): 게이트웨이 구현

- 선택 정책(초기):
  - healthy replica 우선
  - in-flight 요청 수가 가장 적은 replica 우선
  - 동률이면 round-robin
- admission control:
  - 전체 큐 길이 제한(예: 32)
  - replica별 in-flight 제한(기본 1)
  - 초과 시 429로 빠른 실패

M3.3 (0.5~1d): 스트리밍 TTFT 측정 포인트 확정

- 최소 2개 지표를 병행 저장:
  - `t_first_token`: 첫 토큰 생성
  - `t_first_byte_sent`: 첫 chunk flush(`writeStreamStartChunk`/`writeStreamChunk` 포함)

M3.4 (0.5~1d): 운영 자동화(최소)

- healthcheck 주기/타임아웃/서킷 브레이커(일시적 제외)
- 프로세스 관리(PID 파일 또는 systemd)

### 2.4 완료 기준(정량)

- 8노드를 (8노드 TP 1개) 대신 (4노드 TP 2개 replica)로 운영했을 때:
  - 단일 요청 TTFT p95: 4노드 단일 그룹 대비 +10% 이내
  - concurrency=2에서 tokens/s가 1.6x 이상
  - 1000회 연속 요청에서 오류 없이 동작(게이트웨이 포함)

---

## 3. Phase 4 (Week 2~6): PP(Pipeline Parallelism) 구현

Phase 4의 목적은 TP=8로 무리해서 TTFT가 폭발하는 상황을 피하면서, 더 많은 노드를 “단일 요청”에 활용(주로 메모리 fit/throughput)할 수 있도록 **PP=2 x TP=4** 같은 구성을 가능하게 하는 것이다. 기본값은 항상 기존 동작(=PP=1, TP-only)과 호환이어야 한다.

### 3.1 초기 타겟 토폴로지

- TotalNodes=8, `PP=2`, `TP=4`
- Stage 0: 노드 0..3, Stage 1: 노드 4..7
- Slice-preserving: tp_rank i -> 다음 stage의 tp_rank i로 activation slice를 그대로 전달

### 3.2 마일스톤 + 산출물

M4.1 (1~2d): Topology 추상화

- 산출물: `NnParallelTopology` (새 파일)
- 요구사항:
  - PP=1이면 현재 동작 100% 유지
  - nodeIndex -> (pp_rank, tp_rank, global_rank) 계산

M4.2 (2~4d): TP 그룹 범위로 collective/sync 스코프 제한

- 목표: `SYNC_NODE_SLICES`를 "TP 그룹" 내부에서만 수행
- 산출물: `NnNetworkNodeSynchronizer`가 topology 기반 peer 계산
- 검증: PP=1 회귀 0

M4.3 (3~5d): Stage boundary 통신

- 산출물: activation send/recv 모듈(헤더 + 구현)
- 요구사항:
  - 헤더(seq 정보, dtype, payload bytes, slice_id)
  - 타임아웃/오류 처리(데드락 방지)

M4.4 (2~4d): LLM 그래프 분할

- 산출물: `buildLlmNet()`/동등 지점에서 pp_rank별 레이어 범위 선택
- 요구사항:
  - Stage 0: embedding + 첫 절반 레이어, activation 전송
  - Stage 1: activation 수신 후 나머지 레이어 + logits

터치포인트(예):

- `distributed-llama/src/llm.cpp`

M4.5 (2~4d): Prefill chunking + 파이프라인 스케줄링

- 산출물: prefill을 청크로 나눠 stage 간 overlap
- 요구사항:
  - decode는 우선 정확성 우선(초기에는 단순 순차)
  - 긴 prefill 워크로드에서만 chunking 활성화 정책

M4.6 (2~4d): A/B 벤치마크 + 튜닝 리포트

- 대조군: TP=8
- 실험군: PP=2 x TP=4
- 합격 기준:
  - TTFT p95: PP=2xTP=4가 TP=8 대비 개선 또는 최소 동등
  - 안정성: 1000회 연속 요청에서 오류 없음
  - 회귀: PP=1 모드가 기존과 동일

### 3.3 리스크/완화

- 대역폭 병목: chunking + dtype 축소(fp16) + 분할점 튜닝
- 데드락: 시퀀스 번호 + 타임아웃 + 디버그 로그
- stage 불균형: FLOPs 기반 분할점 조정(프로파일러 기반)
- 호환성: PP=1 default 유지 + CI 테스트 바이너리로 회귀 보장

---

## 4. Phase 3~4 실행 순서(권장)

1) Phase 3 옵션 A(외부 라우터)로 replica 운영을 먼저 안정화(빠른 ROI)
2) Phase 4(PP=2xTP=4)로 "더 많은 노드를 단일 요청에 사용"하는 경로를 연다
3) 필요 시 Phase 3 옵션 B(`dllama-api` 내부 replica pool)로 통합/편의성 개선

---

## 5. Definition of Done (공통 체크리스트)

- 빌드/회귀 테스트 통과(CI와 동일)
- 문서 갱신(운영 커맨드/토폴로지/플래그)
- 성능 리포트(Phase 3: replica scaling, Phase 4: PP A/B) 각 1회 이상 작성

---

## Appendix: 작업용 인덱스(참조 경로)

- CI/빌드/테스트: `distributed-llama/Makefile`, `distributed-llama/.github/workflows/main.yml`, `distributed-llama/AGENTS.md`
- API 서버: `distributed-llama/src/dllama-api.cpp`
- CLI 파싱: `distributed-llama/src/app.cpp`
- 통신/collective: `distributed-llama/src/nn/nn-network.cpp`
- 분산 정책 설명: `distributed-llama/NODE_DISTRIBUTION_GUIDE.md`
