# Mixture of Experts (MoE) 가이드

이 문서는 Transformer 기반 LLM에서 사용되는 Mixture of Experts (MoE) 아키텍처에 대한 상세한 설명을 제공합니다.

## 목차

1. [개요](#개요)
2. [MoE의 개념](#moe의-개념)
3. [MoE 구조](#moe-구조)
4. [Gate 메커니즘](#gate-메커니즘)
5. [Expert 계산](#expert-계산)
6. [구현 상세](#구현-상세)
7. [분산 처리](#분산-처리)
8. [장단점](#장단점)

---

## 개요

Mixture of Experts (MoE)는 모델의 파라미터 수를 크게 늘리지 않으면서도 표현력을 향상시키는 기법입니다. 여러 개의 "Expert" 네트워크를 두고, 각 입력에 대해 적절한 Expert를 선택하여 계산합니다.

### 특징

- **효율성**: 모든 Expert를 계산하지 않고 선택된 Expert만 계산
- **확장성**: Expert 수를 늘려 모델 용량 확대 가능
- **전문성**: 각 Expert가 특정 패턴이나 도메인에 특화
- **동적 라우팅**: 입력에 따라 다른 Expert 선택

---

## MoE의 개념

### 기본 아이디어

일반적인 Feed Forward 블록은 모든 입력에 대해 동일한 가중치를 사용합니다. MoE는 여러 개의 Feed Forward 네트워크(Expert)를 두고, 각 입력에 대해 적절한 Expert를 선택합니다.

### 구조 비교

#### 일반 Feed Forward
```
입력 → 단일 FF 블록 → 출력
```

#### MoE Feed Forward
```
입력 → Gate (Expert 선택) → 선택된 Expert들 → 가중 합산 → 출력
```

### 장점

1. **계산 효율성**: 전체 Expert 대신 Top-K Expert만 계산
2. **모델 확장성**: Expert 수를 늘려 모델 용량 확대
3. **전문성**: 각 Expert가 특정 패턴 학습에 특화
4. **동적 처리**: 입력에 따라 다른 Expert 활용

---

## MoE 구조

### 전체 흐름

```
입력
    ↓
Gate 행렬 곱셈 (Expert 점수 계산)
    ↓
Softmax (확률 분포)
    ↓
Top-K Expert 선택
    ↓
선택된 Expert들에 대해:
    - W1, W3 행렬 곱셈
    - SiLU + 곱셈 (비선형 변환)
    - W2 행렬 곱셈
    ↓
Expert 출력에 Gate 가중치 적용
    ↓
Expert 출력 합산
    ↓
출력
```

### 코드 구조

```cpp
// distributed-llama/src/llm.cpp:425-499
// 1. Gate 계산
ff.addOp(OP_MATMUL, "block_moe_gate", ...);
ff.addOp(OP_SOFTMAX, "block_moe_softmax", ...);
ff.addOp(OP_MOE_GATE, "block_moe_gate2", ...);  // Top-K 선택

// 2. Expert 계산 (선택된 Expert만)
ff.addOp(OP_MATMUL, "block_matmul_w1", ...);  // 모든 Expert에 대해
ff.addOp(OP_MATMUL, "block_matmul_w3", ...);
ff.addOp(OP_SILU, "block_act", ...);
ff.addOp(OP_MUL, "block_mul", ...);
ff.addOp(OP_MATMUL, "block_matmul_w2", ...);

// 3. 가중치 적용 및 합산
ff.addOp(OP_SCALE, "block_moe_scale", ...);     // Gate 가중치 적용
ff.addOp(OP_MERGE_SUM, "block_moe_merge_sum", ...);  // Expert 출력 합산
```

---

## Gate 메커니즘

### Gate 행렬 곱셈

#### 수학적 표현

```
gate_scores = XW_gate  (shape: [batch, seq_len, dim] × [dim, nExperts] → [batch, seq_len, nExperts])
```

#### 코드

```cpp
// distributed-llama/src/llm.cpp:433-437
ff.addOp(
    OP_MATMUL, "block_moe_gate", layerIndex,
    pointerBatchConfig(SRC_BUFFER, yBufferIndex),
    pointerBatchConfig(SRC_BUFFER, moeGtBufferIndex),
    n.moeGateSize,
    NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});
```

#### 차원 설명

- **입력 X**: `[batch, seq_len, dim]`
- **가중치 W_gate**: `[dim, nExperts]`
- **출력 gate_scores**: `[batch, seq_len, nExperts]`
  - 각 토큰 위치에서 각 Expert에 대한 점수

### Softmax

```cpp
// distributed-llama/src/llm.cpp:438-443
ff.addOp(
    OP_SOFTMAX, "block_moe_softmax", layerIndex,
    pointerBatchConfig(SRC_BUFFER, moeGtBufferIndex),
    pointerBatchConfig(SRC_BUFFER, moeGtBufferIndex),
    size0(),
    NnSoftmaxOpCodeConfig{});
```

- **역할**: Gate 점수를 확률 분포로 변환
- **수식**: `softmax(x_i) = exp(x_i) / Σ exp(x_j)`
- **효과**: 각 Expert에 대한 선택 확률 생성

### Top-K Expert 선택

```cpp
// distributed-llama/src/llm.cpp:444-449
ff.addOp(
    OP_MOE_GATE, "block_moe_gate2", layerIndex,
    pointerBatchConfig(SRC_BUFFER, moeGtBufferIndex),
    pointerBatchConfig(SRC_BUFFER, moeSBufferIndex),
    size0(),
    NnMoeGateOpCodeConfig{h->nActiveExperts, 1u, moeExpertIndexesBufferIndex});
```

- **역할**: Top-K Expert 선택 및 인덱스 생성
- **파라미터**: `nActiveExperts` (일반적으로 1 또는 2)
- **출력**: 선택된 Expert 인덱스와 가중치

---

## Expert 계산

### Expert 구조

각 Expert는 일반적인 Feed Forward 블록과 동일한 구조를 가집니다:

1. **W1, W3 행렬 곱셈** (Up Projection)
2. **SiLU + 곱셈** (비선형 변환)
3. **W2 행렬 곱셈** (Down Projection)

### 차원

#### 일반 Feed Forward
```
W1, W3: [dim, ffDim]
W2: [ffDim, dim]
```

#### MoE Expert
```
W1, W3: [nExperts, dim, ffDim]  (3D 텐서)
W2: [nExperts, ffDim, dim]       (3D 텐서)
```

각 Expert는 독립적인 가중치를 가집니다.

### 코드

```cpp
// distributed-llama/src/llm.cpp:450-487
// W1 행렬 곱셈 (모든 Expert에 대해)
ff.addOp(
    OP_MATMUL, "block_matmul_w1", layerIndex,
    pointerBatchConfig(SRC_BUFFER, moeYqBufferIndex),
    pointerBatchConfig(SRC_BUFFER, moeDBufferIndex),
    size3D(h->weightType, h->nExperts, n.w1Slice.n, n.w1Slice.d0),
    NnMatmulOpConfig{h->nExperts, h->nActiveExperts, moeExpertIndexesBufferIndex});

// W3 행렬 곱셈
ff.addOp(OP_MATMUL, "block_matmul_w3", ...);

// 비선형 변환
ff.addOp(OP_SILU, "block_act", ...);
ff.addOp(OP_MUL, "block_mul", ...);

// W2 행렬 곱셈
ff.addOp(
    OP_MATMUL, "block_matmul_w2", layerIndex,
    pointerBatchConfig(SRC_BUFFER, moeDQBufferIndex),
    pointerBatchConfig(SRC_BUFFER, moeYBufferIndex),
    size3D(h->weightType, h->nExperts, n.w2Slice.n0, n.w2Slice.d),
    NnMatmulOpConfig{h->nExperts, h->nActiveExperts, moeExpertIndexesBufferIndex});
```

### 효율성

- **선택적 계산**: `NnMatmulOpConfig`의 `nActiveExperts` 파라미터로 선택된 Expert만 계산
- **메모리 효율**: 모든 Expert의 출력을 저장하되, 선택된 Expert만 활성화
- **계산량 감소**: 전체 Expert 대신 Top-K Expert만 계산

---

## 구현 상세

### 1. 입력 준비

```cpp
// distributed-llama/src/llm.cpp:426-431
ff.addOp(
    OP_REPEAT_Z, "block_moe_y_repeat", layerIndex,
    pointerBatchConfig(SRC_BUFFER, yBufferIndex),
    pointerBatchConfig(SRC_BUFFER, moeYqBufferIndex),
    size0(),
    NnRepeatZOpCodeConfig{});
```

- **역할**: 입력을 Expert 수만큼 복제
- **차원 변화**: `[batch, seq_len, dim]` → `[nActiveExperts, batch, seq_len, dim]`

### 2. Gate 계산 및 선택

```cpp
// Gate 점수 계산
ff.addOp(OP_MATMUL, "block_moe_gate", ...);      // [batch, seq_len, nExperts]
ff.addOp(OP_SOFTMAX, "block_moe_softmax", ...);  // 확률 분포
ff.addOp(OP_MOE_GATE, "block_moe_gate2", ...);   // Top-K 선택
```

### 3. Expert 계산

```cpp
// 모든 Expert에 대해 계산 (하지만 선택된 것만 활성화)
ff.addOp(OP_MATMUL, "block_matmul_w1", ...);  // [nExperts, batch, seq_len, ffDim]
ff.addOp(OP_MATMUL, "block_matmul_w3", ...);
ff.addOp(OP_SILU, "block_act", ...);
ff.addOp(OP_MUL, "block_mul", ...);
ff.addOp(OP_MATMUL, "block_matmul_w2", ...);  // [nExperts, batch, seq_len, dim]
```

### 4. 가중치 적용 및 합산

```cpp
// distributed-llama/src/llm.cpp:488-499
// Gate 가중치 적용
ff.addOp(
    OP_SCALE, "block_moe_scale", layerIndex,
    pointerBatchConfig(SRC_BUFFER, moeYBufferIndex),
    pointerBatchConfig(SRC_BUFFER, moeYBufferIndex),
    size0(),
    NnScaleOpCodeConfig{moeSBufferIndex});

// Expert 출력 합산
ff.addOp(
    OP_MERGE_SUM, "block_moe_merge_sum", layerIndex,
    pointerBatchConfig(SRC_BUFFER, moeYBufferIndex),
    pointerBatchConfig(SRC_BUFFER, yBufferIndex),
    size0(),
    NnMergeSumOpCodeConfig{});
```

- **SCALE**: 각 Expert 출력에 Gate 가중치 곱하기
- **MERGE_SUM**: 모든 Expert 출력을 합산하여 최종 출력 생성

---

## 분산 처리

### Gate 행렬 곱셈

```cpp
// distributed-llama/src/llm.cpp:163
n.moeGateSize = size2D(F_32, h->dim, h->nExperts);
```

- **일반적으로**: 모든 노드에서 동일한 Gate 계산
- **분산 불필요**: Gate는 작은 행렬이므로 분산하지 않을 수 있음

### Expert 행렬 곱셈

#### W1, W3 (Up Projection)

```cpp
// distributed-llama/src/llm.cpp:173-175
n.w1Slice = sliceRowMatmul(h->weightType, nNodes, h->dim, ffDim);
n.w3Slice = sliceRowMatmul(h->weightType, nNodes, h->dim, ffDim);
```

- **방식**: Row Matmul (출력 차원 분할)
- **차원**: `[nExperts, dim, ffDim]` → 각 노드는 `[nExperts, dim, ffDim/nNodes]`
- **통신**: All-Gather

#### W2 (Down Projection)

```cpp
// distributed-llama/src/llm.cpp:174
n.w2Slice = sliceColMatmul(h->weightType, nNodes, ffDim, h->dim);
```

- **방식**: Col Matmul (입력 차원 분할)
- **차원**: `[nExperts, ffDim, dim]` → 각 노드는 `[nExperts, ffDim/nNodes, dim]`
- **통신**: All-Reduce

### Expert 선택의 분산 처리

- **동일한 선택**: 모든 노드가 동일한 Expert를 선택해야 함
- **동기화**: Gate 계산 결과를 노드 간 공유
- **일관성**: 모든 노드가 같은 Expert 인덱스 사용

---

## 장단점

### 장점

1. **계산 효율성**
   - 전체 Expert 대신 Top-K Expert만 계산
   - 계산량: `O(nExperts)` → `O(nActiveExperts)`

2. **모델 확장성**
   - Expert 수를 늘려 모델 용량 확대
   - 파라미터 수는 증가하지만 계산량은 제한적

3. **전문성**
   - 각 Expert가 특정 패턴이나 도메인에 특화
   - 더 나은 표현력

4. **동적 처리**
   - 입력에 따라 다른 Expert 선택
   - 입력별 최적화된 처리

### 단점

1. **메모리 사용량**
   - 모든 Expert의 가중치를 저장해야 함
   - Expert 수에 비례하여 메모리 증가

2. **불균형 부하**
   - 인기 있는 Expert에 부하 집중
   - 일부 Expert만 자주 사용될 수 있음

3. **학습 복잡도**
   - Gate와 Expert를 동시에 학습해야 함
   - 수렴이 어려울 수 있음

4. **동기화 오버헤드**
   - 분산 환경에서 Expert 선택 동기화 필요
   - 통신 오버헤드 발생

---

## MoE 파라미터

### 주요 파라미터

```cpp
// distributed-llama/src/llm.hpp
NnUint nExperts;        // 전체 Expert 수
NnUint nActiveExperts;   // 활성화되는 Expert 수 (Top-K)
NnUint moeHiddenDim;     // MoE Expert의 내부 차원
```

### 일반적인 설정

- **nExperts**: 8, 16, 32, 64 등
- **nActiveExperts**: 1 또는 2 (Top-1 또는 Top-2)
- **moeHiddenDim**: 일반적으로 `hiddenDim`과 동일하거나 더 큼

### 예시 (Mixtral 8x7B)

- **nExperts**: 8
- **nActiveExperts**: 2 (Top-2)
- **moeHiddenDim**: 14336
- **효과**: 8개 Expert 중 2개만 계산하여 효율성 향상

---

## 요약

### 핵심 포인트

1. **구조**: Gate → Expert 선택 → Expert 계산 → 가중 합산
2. **효율성**: Top-K Expert만 계산하여 계산량 감소
3. **확장성**: Expert 수를 늘려 모델 용량 확대
4. **전문성**: 각 Expert가 특정 패턴에 특화
5. **분산 처리**: Expert 행렬 곱셈을 노드 간 분산

### 연산 순서

```
입력 [batch, seq_len, dim]
    ↓ Gate
Gate 점수 [batch, seq_len, nExperts]
    ↓ Softmax + Top-K
선택된 Expert 인덱스
    ↓ Expert 계산
Expert 출력 [nActiveExperts, batch, seq_len, dim]
    ↓ 가중치 적용 + 합산
출력 [batch, seq_len, dim]
```

---

## 참고 자료

- 코드 위치: `distributed-llama/src/llm.cpp:425-499`
- Feed Forward 가이드: `FEED_FORWARD_GUIDE.md`
- 행렬 곱셈 가이드: `MATRIX_MULTIPLICATION_GUIDE.md`

---

**작성일**: 2024  
**버전**: 1.0

