# 프롬프트 → 행렬 → 분산 처리 가이드

이 문서는 **사용자가 입력한 프롬프트가 어떻게 행렬(텐서)로 변환되고**, 그 행렬이 **현재 코드 구조에서 어떻게 분산 처리되는지**를 단계별로 길게 설명합니다. 문서 내용은 `distributed-llama/src/llm.cpp`의 네트워크 구성과 연산 흐름을 기준으로 작성했습니다.

---

## 1. 전체 흐름 개요

```
텍스트 프롬프트
  ↓ (토큰화)
토큰 ID 시퀀스
  ↓ (Embedding)
X: [batch, seq_len, dim]
  ↓ (레이어 반복)
[Attention → Feed Forward] × nLayers
  ↓
최종 정규화
  ↓
Logits: [batch, seq_len, vocabSize]
```

**현재 구현은 Tensor Parallelism 방식**을 사용합니다.
- **Row Matmul**: 출력 차원을 분할 (Q/K/V, W1, W3)
- **Col Matmul**: 입력 차원을 분할 (WO, W2, WCLS)
- 각 레이어 끝에서 **SYNC_NODE_SLICES**로 All‑Reduce/All‑Gather 수행

---

## 2. 프롬프트 → 토큰 ID

### 2.1 토큰화

프롬프트는 먼저 **토큰화(Tokenization)** 됩니다.
- 예: `"Hello world"` → `[314, 1923]`
- 이 과정은 **모델 외부에서 수행**되며, 최종적으로 토큰 ID 배열이 네트워크 입력으로 전달됩니다.

### 2.2 토큰 입력 파이프

코드에서 토큰 ID는 `TOK` 파이프를 통해 전달됩니다.

```cpp
// distributed-llama/src/llm.cpp:191-193
n.tokenPipeIndex = netBuilder.addPipe("TOK", size2D(F_32, nBatches, 1));
```

- `TOK` 파이프는 각 배치마다 토큰 ID를 전달
- 이후 `OP_EMBEDDING`에서 임베딩 벡터로 변환

---

## 3. 토큰 ID → 임베딩 행렬

### 3.1 임베딩 연산

```cpp
// distributed-llama/src/llm.cpp:247-254
start.addOp(
    OP_EMBEDDING, "embedding", 0,
    pointerBatchConfig(SRC_PIPE, n.tokenPipeIndex),
    pointerBatchConfig(SRC_PIPE, n.xPipeIndex),
    n.tokenEmbeddingSize,
    NnEmbeddingOpConfig{});
```

- 입력: 토큰 ID (`TOK` 파이프)
- 출력: 임베딩 행렬 `X`

### 3.2 임베딩 행렬 차원

```
X: [batch, seq_len, dim]
```

- **batch**: 동시에 처리하는 입력 개수
- **seq_len**: 시퀀스 길이 (토큰 수)
- **dim**: 모델의 기본 차원 (Hidden Dimension)

임베딩은 **각 토큰을 `dim`차원의 벡터로 변환**합니다.

---

## 4. Attention 블록의 분산 처리

### 4.1 Q/K/V 행렬 곱셈 (Row Matmul)

#### 수식
```
Q = XW_q  (shape: [batch, seq_len, dim] × [dim, qDim])
K = XW_k  (shape: [batch, seq_len, dim] × [dim, kvDim])
V = XW_v  (shape: [batch, seq_len, dim] × [dim, kvDim])
```

#### 분산 방식: Row Matmul (출력 분할)

```cpp
// distributed-llama/src/llm.cpp:168-170
n.qSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->qDim);
n.kSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->kvDim);
n.vSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->kvDim);
```

- 출력 차원(`qDim`, `kvDim`)을 노드 수로 분할
- 각 노드는 **출력의 일부만 계산**

#### 예시 (nNodes=4)

```
X: [batch, seq_len, dim=4096]
W_q: [4096, qDim=4096]

노드0: Q[:, :, 0:1024]
노드1: Q[:, :, 1024:2048]
노드2: Q[:, :, 2048:3072]
노드3: Q[:, :, 3072:4096]
```

결과는 **All‑Gather**로 합쳐서 전체 Q를 구성합니다.

---

### 4.2 KV Cache와 Multi‑Head Attention

```cpp
// distributed-llama/src/llm.cpp:364-384
OP_SHIFT (K, V) → KV Cache
OP_MULTIHEAD_ATT (Q, K, V, KV Cache)
```

- 각 레이어마다 KV Cache가 존재
- KV Cache도 **노드별로 분할 저장**

```cpp
// distributed-llama/src/nn/nn-core.cpp:213-218
kvDim0 = kvDim / nNodes
keySize = [seq_len, kvDim0]
valueSize = [seq_len, kvDim0]
```

즉 **노드마다 KV의 일부만 저장**하며, Multi‑Head Attention 계산 시 그 조각을 사용합니다.

---

### 4.3 WO 행렬 곱셈 (Col Matmul)

#### 수식
```
Y = ZW_o  (shape: [batch, seq_len, qDim] × [qDim, dim])
```

#### 분산 방식: Col Matmul (입력 분할)

```cpp
// distributed-llama/src/llm.cpp:171
n.woSlice = sliceColMatmul(h->weightType, nNodes, h->qDim, h->dim);
```

- 입력 차원(`qDim`)을 노드 수로 분할
- 각 노드는 **입력의 일부만 사용해 부분 결과 생성**

#### 예시 (nNodes=4)

```
Z: [batch, seq_len, qDim=4096]
W_o: [4096, dim=4096]

노드0: Z[:, :, 0:1024] × W_o[0:1024, :]
노드1: Z[:, :, 1024:2048] × W_o[1024:2048, :]
...
```

부분 결과는 **All‑Reduce (합산)** 되어 전체 Y를 만듭니다.

---

## 5. Feed Forward 블록의 분산 처리

### 5.1 W1/W3 (Row Matmul)

```
D = XW_1
L = XW_3
```

- **출력 차원(ffDim)을 분할**
- Row Matmul → All‑Gather

### 5.2 활성화 + 곱셈

```
H = SiLU(D) ⊙ L
```

### 5.3 W2 (Col Matmul)

```
Y = HW_2
```

- **입력 차원(ffDim)을 분할**
- Col Matmul → All‑Reduce

---

## 6. 레이어 간 동기화

```cpp
// distributed-llama/src/llm.cpp:403, 554
att.addSync(zqPipeIndex, SYNC_NODE_SLICES);
ff.addSync(zqPipeIndex, SYNC_NODE_SLICES);
```

- 각 레이어의 출력은 모든 노드가 동일해야 다음 레이어 계산 가능
- All‑Reduce/All‑Gather로 동기화

---

## 7. 최종 Logits 생성

### WCLS (Col Matmul)

```
logits = XW_cls  (shape: [batch, seq_len, dim] × [dim, vocabSize])
```

```cpp
// distributed-llama/src/llm.cpp:587-592
OP_MATMUL "final_matmul_logits"
```

- 입력 차원 `dim` 분할 → Col Matmul
- All‑Reduce로 전체 logits 생성

---

## 8. 요약: 프롬프트가 분산 처리되는 경로

```
프롬프트
  ↓ (토큰화)
토큰 ID
  ↓ (Embedding)
X [batch, seq_len, dim]
  ↓ Attention (Q/K/V: Row Matmul, WO: Col Matmul)
  ↓ Feed Forward (W1/W3: Row Matmul, W2: Col Matmul)
  ↓ 반복 (nLayers)
  ↓ Logits (WCLS: Col Matmul)
```

**분산 핵심 포인트**
- Row Matmul = 출력 분할 + All‑Gather
- Col Matmul = 입력 분할 + All‑Reduce
- 각 레이어 끝에 동기화

---

## 참고 코드 위치

- 네트워크 구성: `distributed-llama/src/llm.cpp` (buildLlmNet)
- KV Cache 슬라이싱: `distributed-llama/src/nn/nn-core.cpp` (sliceKvCache)
- Row/Col Matmul 슬라이싱: `distributed-llama/src/nn/nn-core.cpp`
- 동기화 구현: `distributed-llama/src/nn/nn-network.cpp`

---

**작성일**: 2024  
**버전**: 1.0
