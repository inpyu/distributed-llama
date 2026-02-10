# LLM 행렬 곱셈 가이드

이 문서는 분산 LLM 구현에서 사용되는 행렬 곱셈 연산들을 설명합니다.

## 목차

1. [개요](#개요)
2. [Attention 블록의 행렬 곱셈](#attention-블록의-행렬-곱셈)
3. [Feed-Forward 블록의 행렬 곱셈](#feed-forward-블록의-행렬-곱셈)
4. [최종 출력 행렬 곱셈](#최종-출력-행렬-곱셈)
5. [MoE (Mixture of Experts)](#moe-mixture-of-experts)
6. [분산 처리 방식](#분산-처리-방식)
7. [LLM에서의 영향](#llm에서의-영향)

---

## 개요

Transformer 기반 LLM은 여러 행렬 곱셈 연산으로 구성됩니다. 각 연산은 특정 역할을 수행하며, 분산 처리 환경에서는 행렬을 노드 간에 분할하여 병렬로 계산합니다.

### 주요 행렬 곱셈 유형

- **Row Matmul**: 출력 차원을 분할 (Q, K, V, W1, W3)
- **Col Matmul**: 입력 차원을 분할 (WO, W2, WCLS)

---

## Attention 블록의 행렬 곱셈

### 1. Q, K, V 행렬 곱셈 (Query, Key, Value Projection)

#### 수학적 표현

```
Q = XW_q  (shape: [batch, seq_len, dim] × [dim, qDim] → [batch, seq_len, qDim])
K = XW_k  (shape: [batch, seq_len, dim] × [dim, kvDim] → [batch, seq_len, kvDim])
V = XW_v  (shape: [batch, seq_len, dim] × [dim, kvDim] → [batch, seq_len, kvDim])
```

#### 차원 설명

**입력 텐서 X의 차원: `[batch, seq_len, dim]`**

- **`batch`**: 배치 크기 - 동시에 처리하는 입력 시퀀스의 개수
  - 예: `batch=4` → 4개의 서로 다른 입력을 동시에 처리
  - 배치 처리로 GPU 활용도 향상 및 처리량 증가

- **`seq_len`**: 시퀀스 길이 - 각 입력 시퀀스의 토큰 개수
  - 예: `seq_len=2048` → 각 입력이 2048개의 토큰으로 구성
  - 모델의 최대 컨텍스트 길이에 의해 제한됨

- **`dim`**: 모델의 기본 차원 (Hidden Dimension)
  - 예: `dim=4096` → 각 토큰이 4096차원의 벡터로 표현됨
  - 모델의 표현력과 용량을 결정하는 핵심 파라미터
  - 모든 레이어에서 일관되게 사용되는 차원

**가중치 행렬 W_k, W_v의 차원: `[dim, kvDim]`**

- **`dim`**: 입력 차원 - 입력 텐서의 마지막 차원과 일치해야 함
- **`kvDim`**: Key/Value 차원 - 어텐션 계산에 사용될 차원
  - `kvDim = headDim × nKvHeads`
  - `headDim`: 각 어텐션 헤드의 차원 (예: 128)
  - `nKvHeads`: Key/Value 헤드의 개수 (GQA - Grouped Query Attention 사용 시 `nHeads`보다 작을 수 있음)
  - 예: `headDim=128`, `nKvHeads=8` → `kvDim=1024`

**출력 텐서 K, V의 차원: `[batch, seq_len, kvDim]`**

#### 행렬 곱셈 차원 계산 규칙

행렬 곱셈에서 차원은 다음과 같이 계산됩니다:

```
[batch, seq_len, dim] × [dim, kvDim] = [batch, seq_len, kvDim]
     ↑      ↑      ↑        ↑   ↑         ↑      ↑      ↑
   유지   유지   곱셈    곱셈  유지      유지   유지   결과
```

**계산 과정:**

1. **앞 차원 유지**: `batch`와 `seq_len`은 그대로 유지됩니다
   - 각 배치의 각 토큰 위치에 대해 독립적으로 행렬 곱셈 수행

2. **마지막 두 차원 곱셈**: `[..., dim] × [dim, kvDim]`
   - 입력의 마지막 차원(`dim`)과 가중치의 첫 번째 차원(`dim`)이 일치해야 함
   - 결과의 마지막 차원은 가중치의 두 번째 차원(`kvDim`)이 됨

3. **수학적 표현**:
   ```
   K[b, s, :] = X[b, s, :] × W_k
   ```
   - `b`: 배치 인덱스 (0 ~ batch-1)
   - `s`: 시퀀스 위치 인덱스 (0 ~ seq_len-1)
   - 각 `[b, s]` 위치에서 `[dim]` 벡터와 `[dim, kvDim]` 행렬을 곱하여 `[kvDim]` 벡터 생성

**왜 `[batch, seq_len, kvDim]`이 나오는가?**

- **배치 차원 유지**: 모든 배치 항목을 병렬로 처리하므로 배치 차원 유지
- **시퀀스 차원 유지**: 각 토큰 위치마다 독립적으로 변환하므로 시퀀스 차원 유지
- **차원 변환**: 입력의 `dim` 차원을 어텐션에 필요한 `kvDim` 차원으로 변환
  - `kvDim`은 어텐션 헤드 수와 헤드 차원에 의해 결정됨
  - GQA (Grouped Query Attention)를 사용하는 경우 `kvDim < qDim`일 수 있음

**실제 예시:**

```
입력 X: [4, 2048, 4096]
  - batch=4: 4개의 입력 시퀀스
  - seq_len=2048: 각 시퀀스는 2048개 토큰
  - dim=4096: 각 토큰은 4096차원 벡터

가중치 W_k: [4096, 1024]
  - dim=4096: 입력 차원과 일치
  - kvDim=1024: Key/Value 차원 (headDim=128, nKvHeads=8)

출력 K: [4, 2048, 1024]
  - batch=4: 배치 차원 유지
  - seq_len=2048: 시퀀스 차원 유지
  - kvDim=1024: 변환된 차원
```

**차원 관계:**

```cpp
// distributed-llama/src/llm.cpp:106-109
header.qDim = header.headDim * header.nHeads;      // Query 차원
header.kvDim = header.headDim * header.nKvHeads;   // Key/Value 차원
```

- `qDim`: Query 헤드 수 × 헤드 차원
- `kvDim`: Key/Value 헤드 수 × 헤드 차원
- GQA를 사용하면 `nKvHeads < nHeads`이므로 `kvDim < qDim`

#### 코드 위치

```cpp
// distributed-llama/src/llm.cpp:303-320
att.addOp(
    OP_MATMUL, "block_matmul_q", layerIndex,
    pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
    pointerBatchConfig(SRC_BUFFER, qBufferIndex),
    size2D(h->weightType, n.qSlice.n, n.qSlice.d0),
    NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});
```

#### 역할

- **입력 변환**: 레이어의 입력 텐서를 Query, Key, Value로 변환
- **어텐션 계산 준비**: 이후 Multi-Head Attention 연산에서 사용
- **차원 변환**: 입력 차원(`dim`)을 어텐션 차원(`qDim`, `kvDim`)으로 변환

#### 분산 처리

```cpp
// distributed-llama/src/llm.cpp:168-170
n.qSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->qDim);
n.kSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->kvDim);
n.vSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->kvDim);
```

- **방식**: Row Matmul (출력 차원 분할)
- **분할**: 각 노드는 `d0 = d/nNodes` 크기의 부분 결과 생성
- **예시**: `dim=4096`, `nNodes=4` → 각 노드는 `1024` 차원의 Q/K/V 생성
- **후처리**: All-Gather로 전체 결과 수집

---

### 2. WO 행렬 곱셈 (Output Projection)

#### 수학적 표현

```
Y = ZW_o  (shape: [batch, seq_len, qDim] × [qDim, dim] → [batch, seq_len, dim])
```

여기서 `Z`는 Multi-Head Attention의 출력입니다.

#### 코드 위치

```cpp
// distributed-llama/src/llm.cpp:391-396
att.addOp(
    OP_MATMUL, "block_matmul_wo", layerIndex,
    pointerBatchConfig(SRC_BUFFER, zqSliceBufferIndex),
    pointerBatchConfig(SRC_BUFFER, yBufferIndex),
    size2D(h->weightType, n.woSlice.n0, n.woSlice.d),
    NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});
```

#### 역할

- **차원 복원**: 어텐션 출력을 원래 입력 차원으로 투영
- **정보 통합**: 여러 헤드의 어텐션 결과를 통합
- **Residual Connection 준비**: Feed-Forward 블록으로 전달하기 전 차원 맞춤

#### 분산 처리

```cpp
// distributed-llama/src/llm.cpp:171
n.woSlice = sliceColMatmul(h->weightType, nNodes, h->qDim, h->dim);
```

- **방식**: Col Matmul (입력 차원 분할)
- **분할**: 각 노드는 `n0 = n/nNodes` 크기의 부분 결과 생성
- **후처리**: All-Reduce로 부분 결과 합산

---

## Feed-Forward 블록의 행렬 곱셈

### 1. W1, W3 행렬 곱셈 (Up Projection)

#### 수학적 표현

```
D = XW_1  (shape: [batch, seq_len, dim] × [dim, ffDim] → [batch, seq_len, ffDim])
L = XW_3  (shape: [batch, seq_len, dim] × [dim, ffDim] → [batch, seq_len, ffDim])
```

#### 코드 위치

```cpp
// distributed-llama/src/llm.cpp:509-520
ff.addOp(
    OP_MATMUL, "block_matmul_w1", layerIndex,
    pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
    pointerBatchConfig(SRC_BUFFER, dBufferIndex),
    size2D(h->weightType, n.w1Slice.n, n.w1Slice.d0),
    NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});
ff.addOp(
    OP_MATMUL, "block_matmul_w3", layerIndex,
    pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
    pointerBatchConfig(SRC_BUFFER, lBufferIndex),
    size2D(h->weightType, n.w3Slice.n, n.w3Slice.d0),
    NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});
```

#### 역할

- **차원 확장**: 입력 차원(`dim`)을 Feed-Forward 차원(`ffDim`)으로 확장
- **비선형 변환 준비**: SiLU 활성화 함수와 곱셈 연산에 사용
- **표현력 향상**: 더 넓은 차원에서 비선형 변환 수행

#### 처리 흐름

```cpp
// distributed-llama/src/llm.cpp:521-532
ff.addOp(OP_SILU, "block_act", layerIndex, ...);  // SiLU(D)
ff.addOp(OP_MUL, "block_mul", layerIndex, ...);    // SiLU(D) ⊙ L
```

- `D`에 SiLU 활성화 함수 적용
- `D`와 `L`을 요소별 곱셈 (`SiLU(D) ⊙ L`)

#### 분산 처리

```cpp
// distributed-llama/src/llm.cpp:173-175
n.w1Slice = sliceRowMatmul(h->weightType, nNodes, h->dim, ffDim);
n.w3Slice = sliceRowMatmul(h->weightType, nNodes, h->dim, ffDim);
```

- **방식**: Row Matmul (출력 차원 분할)
- **분할**: 각 노드는 `ffDim/nNodes` 크기의 부분 결과 생성
- **후처리**: All-Gather로 전체 결과 수집

---

### 2. W2 행렬 곱셈 (Down Projection)

#### 수학적 표현

```
Y = (SiLU(D) ⊙ L)W_2  (shape: [batch, seq_len, ffDim] × [ffDim, dim] → [batch, seq_len, dim])
```

#### 코드 위치

```cpp
// distributed-llama/src/llm.cpp:541-546
ff.addOp(
    OP_MATMUL, "block_matmul_w2", layerIndex,
    pointerBatchConfig(SRC_BUFFER, dqBufferIndex),
    pointerBatchConfig(SRC_BUFFER, yBufferIndex),
    size2D(h->weightType, n.w2Slice.n0, n.w2Slice.d),
    NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});
```

#### 역할

- **차원 축소**: 확장된 차원(`ffDim`)을 원래 차원(`dim`)으로 복원
- **정보 압축**: 비선형 변환된 정보를 원래 차원으로 압축
- **Residual Connection**: Attention 블록의 출력과 더해짐

#### 분산 처리

```cpp
// distributed-llama/src/llm.cpp:174
n.w2Slice = sliceColMatmul(h->weightType, nNodes, ffDim, h->dim);
```

- **방식**: Col Matmul (입력 차원 분할)
- **분할**: 각 노드는 `ffDim/nNodes` 크기의 부분 입력 처리
- **후처리**: All-Reduce로 부분 결과 합산

---

## 최종 출력 행렬 곱셈

### WCLS 행렬 곱셈 (Logits Projection)

#### 수학적 표현

```
logits = XW_cls  (shape: [batch, seq_len, dim] × [dim, vocabSize] → [batch, seq_len, vocabSize])
```

#### 코드 위치

```cpp
// distributed-llama/src/llm.cpp:587-592
end.addOp(
    OP_MATMUL, "final_matmul_logits", 0,
    pointerBatchedSliceConfig(SRC_BUFFER, yqBufferIndex),
    pointerBatchConfig(SRC_BUFFER, logitsSliceBufferIndex),
    size2D(h->weightType, n.wclsSlice.n0, n.wclsSlice.d),
    NnMatmulOpConfig{});
```

#### 역할

- **어휘 투영**: 마지막 레이어의 출력을 어휘 크기로 변환
- **다음 토큰 예측**: 각 토큰 위치에서 다음 토큰의 확률 분포 생성
- **최종 출력**: 모델의 최종 예측 결과

#### 분산 처리

```cpp
// distributed-llama/src/llm.cpp:176
n.wclsSlice = sliceColMatmul(h->weightType, nNodes, h->dim, h->vocabSize);
```

- **방식**: Col Matmul (입력 차원 분할)
- **분할**: 각 노드는 `dim/nNodes` 크기의 부분 입력 처리
- **후처리**: All-Reduce로 부분 결과 합산하여 전체 어휘 크기의 로짓 생성

---

## MoE (Mixture of Experts)

### Gate 행렬 곱셈

#### 수학적 표현

```
gate_scores = XW_gate  (shape: [batch, seq_len, dim] × [dim, nExperts] → [batch, seq_len, nExperts])
```

#### 코드 위치

```cpp
// distributed-llama/src/llm.cpp:433-437
ff.addOp(
    OP_MATMUL, "block_moe_gate", layerIndex,
    pointerBatchConfig(SRC_BUFFER, yBufferIndex),
    pointerBatchConfig(SRC_BUFFER, moeGtBufferIndex),
    n.moeGateSize,
    NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});
```

#### 역할

- **Expert 선택**: 각 토큰에 대해 활성화할 Expert 결정
- **라우팅**: 입력을 적절한 Expert로 분배
- **효율성**: 전체 Expert를 계산하지 않고 선택된 Expert만 계산

#### 처리 흐름

1. Gate 행렬 곱셈으로 Expert 점수 계산
2. Softmax로 확률 분포 생성
3. Top-K Expert 선택
4. 선택된 Expert만 계산 수행

---

## 분산 처리 방식

### Row Matmul (행 분할)

#### 구현

```cpp
// distributed-llama/src/nn/nn-core.cpp:222-231
NnRowMatmulSlice sliceRowMatmul(NnFloatType type, NnUint nNodes, NnUint n, NnUint d) {
    NnRowMatmulSlice s;
    assert(d % nNodes == 0);
    s.d0 = d / nNodes;  // 출력 차원 분할
    s.n = n;
    s.sliceSize = size2D(type, s.n, s.d0);
    return s;
}
```

#### 특징

- **분할 대상**: 출력 차원 (`d`)
- **각 노드 결과**: `[n, d0]` 크기의 부분 행렬
- **통신**: All-Gather로 전체 결과 수집
- **사용 위치**: Q, K, V, W1, W3

#### 예시

```
전체 행렬: [4096, 8192] (n=4096, d=8192)
노드 수: 4
각 노드: [4096, 2048] (n=4096, d0=2048)
→ All-Gather 후: [4096, 8192]
```

---

### Col Matmul (열 분할)

#### 구현

```cpp
// distributed-llama/src/nn/nn-core.cpp:234-244
NnColMatmulSlice sliceColMatmul(NnFloatType type, NnUint nNodes, NnUint n, NnUint d) {
    NnColMatmulSlice s;
    assert(n % nNodes == 0);
    s.n0 = n / nNodes;  // 입력 차원 분할
    s.d = d;
    s.sliceSize = size2D(type, s.n0, d);
    return s;
}
```

#### 특징

- **분할 대상**: 입력 차원 (`n`)
- **각 노드 결과**: `[n0, d]` 크기의 부분 행렬
- **통신**: All-Reduce로 부분 결과 합산
- **사용 위치**: WO, W2, WCLS

#### 예시

```
전체 행렬: [8192, 4096] (n=8192, d=4096)
노드 수: 4
각 노드: [2048, 4096] (n0=2048, d=4096)
→ All-Reduce (합산) 후: [4096] (출력 차원)
```

---

### 동기화 지점

각 레이어의 행렬 곱셈 후 동기화가 필요합니다:

```cpp
// distributed-llama/src/llm.cpp:403, 554
att.addSync(zqPipeIndex, SYNC_NODE_SLICES);  // Attention 후
ff.addSync(zqPipeIndex, SYNC_NODE_SLICES);   // Feed-Forward 후
```

- **SYNC_NODE_SLICES**: All-Reduce 또는 All-Gather 수행
- **목적**: 노드 간 부분 결과 통합
- **필요성**: 다음 레이어로 전달하기 전 완전한 결과 필요

---

## LLM에서의 영향

### 1. Attention 블록

- **토큰 간 관계 모델링**: Q, K, V를 통해 토큰 간 유사도 계산
- **장거리 의존성**: 시퀀스 전체에서 정보 수집
- **컨텍스트 이해**: 입력 시퀀스의 의미 파악

### 2. Feed-Forward 블록

- **비선형 변환**: 선형 변환만으로는 불가능한 복잡한 패턴 학습
- **표현력 확장**: 더 넓은 차원에서 특징 추출
- **정보 처리**: Attention에서 추출한 정보를 변환 및 압축

### 3. 최종 출력

- **다음 토큰 예측**: 언어 모델링의 핵심 기능
- **확률 분포 생성**: 어휘 전체에 대한 확률 분포 제공
- **생성 제어**: 샘플링 전략에 따라 다음 토큰 선택

### 4. MoE (선택적)

- **계산 효율성**: 전체 Expert 대신 선택된 Expert만 계산
- **모델 확장성**: Expert 수를 늘려 모델 용량 확대
- **전문성**: 각 Expert가 특정 패턴에 특화

---

## 요약

### 행렬 곱셈 매핑

| 연산 | 타입 | 입력 | 출력 | 분산 방식 | 통신 |
|------|------|------|------|-----------|------|
| Q, K, V | Row Matmul | [batch, seq, dim] | [batch, seq, qDim/kvDim] | 출력 분할 | All-Gather |
| WO | Col Matmul | [batch, seq, qDim] | [batch, seq, dim] | 입력 분할 | All-Reduce |
| W1, W3 | Row Matmul | [batch, seq, dim] | [batch, seq, ffDim] | 출력 분할 | All-Gather |
| W2 | Col Matmul | [batch, seq, ffDim] | [batch, seq, dim] | 입력 분할 | All-Reduce |
| WCLS | Col Matmul | [batch, seq, dim] | [batch, seq, vocabSize] | 입력 분할 | All-Reduce |

### 핵심 포인트

1. **Row Matmul**: 출력 차원 분할 → All-Gather 필요
2. **Col Matmul**: 입력 차원 분할 → All-Reduce 필요
3. **각 레이어마다 동기화**: 완전한 결과를 다음 레이어로 전달
4. **순차적 의존성**: 레이어 i의 출력이 레이어 i+1의 입력

---

## 참고 자료

- 코드 위치: `distributed-llama/src/llm.cpp`
- 슬라이싱 함수: `distributed-llama/src/nn/nn-core.cpp`
- 네트워크 동기화: `distributed-llama/src/nn/nn-network.cpp`

---

**작성일**: 2024  
**버전**: 1.0

