# PP 파라미터 실험 결과 (2026-03-03)

본 문서는 재배포 이후 원격 8-worker 환경에서 수행한 `pp-size`/prefill 파라미터 실험 결과를 정리한다.

## 실험 환경

- Cluster: 8 workers (`100.78.3.114:9999`, `100.68.147.68:9999`, `100.77.7.70:9999`, `100.70.41.9:9999`, `100.67.190.3:9999`, `100.84.48.55:9999`, `100.76.95.128:9999`)
- Model: `dllama_model_original_q40.m`
- Tokenizer: `dllama_tokenizer_llama3.t`
- Buffer float type: `q80`
- Threads: `4`
- Steps: `420`
- Collective: `auto`
- net-turbo: `0`
- Prompt: `The quick brown fox jumps over the lazy dog.` 반복 20회

## 실험 매트릭스 및 결과

케이스 이름 규칙:

- `pXa`/`pXb`/`pXc`/`pXd`에서 `X`는 `pp-size`를 의미한다. 예: `p1*`는 `pp-size=1`, `p2*`는 `pp-size=2`.
- 접미사(`a`,`b`,`c`,`d`)는 같은 `pp-size` 내에서 `prefill-chunk-threshold`/`prefill-chunk-size` 조합을 구분하기 위한 실험 인덱스다.
- 실제 파라미터는 로그 파일명(`pp{n}_th{t}_ch{c}`)과 아래 표의 수치를 기준으로 해석한다.

요청한 케이스(`p1a`, `p1b`, `p2a`, `p2b`, `p2c`, `p2d`)의 의미:

- `p1a` = `--pp-size 1 --prefill-chunk-threshold 128 --prefill-chunk-size 0`
- `p1b` = `--pp-size 1 --prefill-chunk-threshold 64 --prefill-chunk-size 0`
- `p2a` = `--pp-size 2 --prefill-chunk-threshold 64 --prefill-chunk-size 0`
- `p2b` = `--pp-size 2 --prefill-chunk-threshold 64 --prefill-chunk-size 8`
- `p2c` = `--pp-size 2 --prefill-chunk-threshold 128 --prefill-chunk-size 0`
- `p2d` = `--pp-size 2 --prefill-chunk-threshold 128 --prefill-chunk-size 8`

| Case | pp-size | prefill-chunk-threshold | prefill-chunk-size | Exit code | Eval tok/s | Pred tok/s | TTFT(ms) | Decode(ms) | Total(ms) | 상태 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| p1a | 1 | 128 | 0 | 0 | 1.35 | 1.24 | 149276.83 | 180201.17 | 328665.22 | 성공 |
| p1b | 1 | 64 | 0 | 1 | - | - | - | - | - | 실패 (`SYNC_NODES` stall -> `Socket closed`) |
| p2a | 2 | 64 | 0 | 0 | 6.50 | 5.73 | 73962.23 | 82718.02 | 153798.02 | 성공 |
| p2b | 2 | 64 | 8 | 1 | - | - | - | - | - | 실패 (`SYNC_NODES` stall -> `Socket closed`) |
| p2c | 2 | 128 | 0 | 0 | 6.61 | 5.77 | 72642.38 | 82499.24 | 152024.19 | 성공 |
| p2d | 2 | 128 | 8 | 1 | - | - | - | - | - | 실패 (`SYNC_NODES` stall -> `Socket closed`) |
| p4a | 4 | 64 | 0 | 0 | 33.79 | 26.29 | 25066.87 | 28741.06 | 53297.24 | 성공 |
| p4b | 4 | 64 | 4 | 0 | 33.32 | 26.37 | 24959.85 | 28985.66 | 53411.62 | 성공 |
| p4c | 4 | 128 | 0 | 0 | 33.52 | 26.30 | 24981.24 | 28801.28 | 53236.07 | 성공 |
| p4d | 4 | 128 | 4 | 0 | 32.37 | 26.41 | 25015.26 | 28657.44 | 53163.56 | 성공 |

## 관찰 요약

"최적(Optimal)" 정의(2026-03-03 baseline 매트릭스, 동일 조건 비교): (1) 안정성(Exit code 0), (2) TTFT/Total 낮음, (3) Pred tok/s 높음.

1. 최적(Optimal, baseline 기준): `p4d` = `--pp-size 4 --prefill-chunk-threshold 128 --prefill-chunk-size 4` (Pred 26.41 tok/s, Total 53163.56 ms, Exit code 0)
2. `pp-size=4` 구간이 성능/안정성 모두 가장 우수하다.
3. `pp-size=2`는 `prefill-chunk-size=0`에서는 안정적으로 완료된다.
4. `pp-size=2`에서 `prefill-chunk-size=8`은 baseline(2026-03-03)에서는 threshold 64/128 모두 실패했으나, retry(2026-03-04)에서 완료 사례가 관측됐다(단, 동일 조건 비교 불가).
5. `pp-size=1`은 완료되더라도 성능이 매우 낮고, 일부 조합에서 초기 `SYNC_NODES` stall이 발생한다.

## 현재 권장 프리셋

아래 프리셋은 2026-03-03 baseline 수치(`net-turbo: 0`) 기준이다. baseline은 동일 조건 비교를 위해 `--net-turbo 0`을 고정한다.

- 최종 결론(Optimal, baseline 기준): `--net-turbo 0 --pp-size 4 --prefill-chunk-threshold 128 --prefill-chunk-size 4` (p4d)
- 대안(baseline 기준, Eval tok/s 약간 높음): `--net-turbo 0 --pp-size 4 --prefill-chunk-threshold 64 --prefill-chunk-size 0` (p4a)
- 참고: `pp-size=2` + `prefill-chunk-size=8`은 baseline에서는 실패했으나 retry에서 완료 사례가 있다. 운영 권장(최적)은 apples-to-apples 비교를 위해 `pp-size=4`를 유지한다.

## 원시 로그 파일

- `postdeploy_p1a_pp1_th128_ch0_20260303_195523.log`
- `postdeploy_p1b_pp1_th64_ch0_20260303_200133.log`
- `postdeploy_p2a_pp2_th64_ch0_20260303_200242.log`
- `postdeploy_p2b_pp2_th64_ch8_20260303_200559.log`
- `postdeploy_p2c_pp2_th128_ch0_20260303_200652.log`
- `postdeploy_p2d_pp2_th128_ch8_20260303_201006.log`
- `postdeploy_p4a_pp4_th64_ch0_20260303_201100.log`
- `postdeploy_p4b_pp4_th64_ch4_20260303_201236.log`
- `postdeploy_p4c_pp4_th128_ch0_20260303_201412.log`
- `postdeploy_p4d_pp4_th128_ch4_20260303_201547.log`

## 재실험 결과 (2026-03-04, 기존 실패 케이스)

동일한 원격 8-worker 클러스터에서 기존 실패 케이스(`p1b`, `p2b`, `p2d`)를 재실행했다.

주의: 아래 retry는 안정성 재확인 목적이며, 2026-03-03 기준 실험과 동일 조건 비교로 성능을 직접 비교하면 안 된다. 기준 섹션은 `Steps: 420`, `Tokenizer: dllama_tokenizer_llama3.t`, `net-turbo: 0`을 명시한다. 재실험 실행 기록상 `--steps 220`, tokenizer `dllama_tokenizer_llama3_8B.t`로 실행됐으며, retry 로그에 `🚁 Network is in non-blocking mode`가 출력돼 net-turbo 관련 설정도 달랐던 것으로 파악된다. retry 로그는 전체 CLI 인자를 출력하지 않으므로, retry 성공은 "기존 실패 케이스가 같은 클러스터에서 재시도 시 완료까지 도달한다"는 안정성 신호로만 해석한다. 원래 매트릭스의 실패 상태를 완전히 뒤집으려면 2026-03-03과 동일 조건으로 재실행해 재현성을 확인해야 한다.

| Case | pp-size | prefill-chunk-threshold | prefill-chunk-size | Exit code | Eval tok/s | Pred tok/s | TTFT(ms) | Decode(ms) | Total(ms) | 상태 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| p1b (retry) | 1 | 64 | 0 | 0 | 1.23 | 1.14 | 163142.00 | 17846.47 | 180126.39 | 성공 (중간 `EXEC_STALL` 로그는 있으나 timeout/Socket closed 없이 완료) |
| p2b (retry) | 2 | 64 | 8 | 0 | 5.67 | 4.96 | 72484.04 | 10089.56 | 79926.08 | 성공 |
| p2d (retry) | 2 | 128 | 8 | 0 | 5.68 | 4.76 | 74686.37 | 10234.20 | 82243.17 | 성공 |

재실험 로그:

- `postdeploy_retry_p1b_pp1_th64_ch0_20260304_165010.log`
- `postdeploy_retry_p2b_pp2_th64_ch8_20260304_165409.log`
- `postdeploy_retry_p2d_pp2_th128_ch8_20260304_165625.log`
