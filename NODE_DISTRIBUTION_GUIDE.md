# 노드별 분산 처리 가이드

이 문서는 현재 코드에서 **노드별로 작업이 어떻게 분담되는지**를 설명합니다. 핵심은 **레이어를 나누는 Pipeline Parallelism이 아니라**, **텐서 차원을 나누는 Tensor Parallelism** 방식이라는 점입니다. KV Cache도 같은 원리로 **차원 분할**됩니다.

---

## 1. 결론 요약

- **레이어를 나누지 않습니다.** 모든 노드는 **모든 레이어를 동일하게 수행**합니다.
- **각 레이어의 연산을 차원 기준으로 분할합니다.**
  - 출력 차원 분할: Row Matmul
  - 입력 차원 분할: Col Matmul
- **KV Cache도 분할됩니다.** 레이어별 KV Cache를 노드 수로 나눠 각 노드가 일부만 저장합니다.
- 레이어마다 **동기화(All‑Gather / All‑Reduce)** 가 발생합니다.

---

## 2. 현재 구조: Tensor Parallelism

### 2.1 레이어 분할이 아닌 이유

현재 코드에서 레이어는 다음과 같이 **모든 노드에서 동일하게 반복**됩니다:

```cpp
// distributed-llama/src/llm.cpp
for (NnUint nodeIndex = 0; nodeIndex < nNodes; nodeIndex++) {
    ...
    for (NnUint layerIndex = 0; layerIndex < h->nLayers; layerIndex++) {
        // 모든 노드가 모든 레이어 구성
    }
}
```

즉, **Pipeline Parallelism(레이어 분할)** 은 사용하지 않습니다.

---

### 2.2 작업 분담 방식

각 노드는 **동일한 연산 그래프**를 수행하지만, **행렬 곱셈의 차원 일부만 계산**합니다.

- **Row Matmul**: 출력 차원을 나눠 계산
- **Col Matmul**: 입력 차원을 나눠 계산

이 방식은 **Tensor Parallelism**으로 분류됩니다.

---

## 3. 노드별 역할: Row Matmul / Col Matmul

### 3.1 Row Matmul (출력 차원 분할)

출력 차원 `d`를 노드 수로 분할합니다.

```cpp
// distributed-llama/src/nn/nn-core.cpp
s.d0 = d / nNodes;  // 출력 차원 분할
```

#### 예시

- `X: [batch, seq_len, dim]`
- `W: [dim, d]`
- 노드 수 4

```
노드0: 출력 [batch, seq_len, d/4]
노드1: 출력 [batch, seq_len, d/4]
노드2: 출력 [batch, seq_len, d/4]
노드3: 출력 [batch, seq_len, d/4]
```

#### 통신
- 각 노드의 출력을 **All‑Gather**로 합쳐 전체 출력 생성

---

### 3.2 Col Matmul (입력 차원 분할)

입력 차원 `n`을 노드 수로 분할합니다.

```cpp
// distributed-llama/src/nn/nn-core.cpp
s.n0 = n / nNodes;  // 입력 차원 분할
```

#### 예시

- `X: [batch, seq_len, n]`
- `W: [n, d]`
- 노드 수 4

```
노드0: X[:, :, 0:n/4] × W[0:n/4, :]
노드1: X[:, :, n/4:2n/4] × W[n/4:2n/4, :]
노드2: X[:, :, 2n/4:3n/4] × W[2n/4:3n/4, :]
노드3: X[:, :, 3n/4:4n/4] × W[3n/4:4n/4, :]
```

#### 통신
- 각 노드의 결과를 **All‑Reduce(합산)** 하여 전체 출력 생성

---

## 4. 실제 LLM에서의 분담 위치

### 4.1 Attention 블록

| 연산 | 분할 방식 | 이유 |
|------|-----------|------|
| Q, K, V | Row Matmul | 출력 차원이 커서 분산 효율 높음 |
| WO | Col Matmul | 입력 차원을 분할하여 메모리 절약 |

### 4.2 Feed Forward 블록

| 연산 | 분할 방식 | 이유 |
|------|-----------|------|
| W1, W3 | Row Matmul | 출력 차원(ffDim) 분할 |
| W2 | Col Matmul | 입력 차원(ffDim) 분할 |

### 4.3 Logits

| 연산 | 분할 방식 | 이유 |
|------|-----------|------|
| WCLS | Col Matmul | 입력 차원(dim) 분할 |

---

## 5. KV Cache 분산 방식

KV Cache는 **레이어별로 존재**하며, **노드 수로 분할**됩니다.

```cpp
// distributed-llama/src/nn/nn-core.cpp
kvDim0 = kvDim / nNodes;
keySize   = [seq_len, kvDim0]
valueSize = [seq_len, kvDim0]
```

즉:
- 각 레이어마다 KV Cache가 있음
- 각 노드는 **자신의 kvDim0 부분만** 저장
- 노드 전체가 모이면 전체 KV Cache가 구성됨

**중요**: KV Cache는 **레이어를 분할한 결과가 아니라, 차원을 분할한 결과**입니다.

---

## 6. 동기화 지점

레이어마다 동기화가 발생합니다.

```cpp
// distributed-llama/src/llm.cpp
att.addSync(zqPipeIndex, SYNC_NODE_SLICES);
ff.addSync(zqPipeIndex, SYNC_NODE_SLICES);
```

- 각 노드는 부분 결과만 가지고 있으므로
- 다음 레이어로 넘어가기 전에 **전체 출력이 필요**
- 따라서 All‑Gather / All‑Reduce 수행

---

## 7. 정리

### 현재 구조의 특징

- ✅ **레이어 분할 아님** (Pipeline Parallelism 없음)
- ✅ **텐서 차원 분할** (Tensor Parallelism)
- ✅ KV Cache도 동일한 방식으로 분할
- ✅ 레이어마다 동기화

### 노드별 작업 분담 요약

```
모든 노드가 모든 레이어 실행
  ↓
각 레이어 내부에서 행렬 곱셈을 차원 분할
  ↓
부분 결과를 통신으로 합산
```

---

## 참고 코드 위치

- 네트워크 구성: `distributed-llama/src/llm.cpp`
- Row/Col Matmul 슬라이싱: `distributed-llama/src/nn/nn-core.cpp`
- KV Cache 슬라이싱: `distributed-llama/src/nn/nn-core.cpp`
- 동기화 구현: `distributed-llama/src/nn/nn-network.cpp`

---

**작성일**: 2024  
**버전**: 1.0
