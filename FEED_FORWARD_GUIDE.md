# Feed Forward 블록 가이드

이 문서는 Transformer 기반 LLM의 Feed Forward (FF) 블록에 대한 상세한 설명을 제공합니다.

## 목차

1. [개요](#개요)
2. [Feed Forward 블록의 구조](#feed-forward-블록의-구조)
3. [수학적 표현](#수학적-표현)
4. [구현 상세](#구현-상세)
5. [행렬 곱셈 연산](#행렬-곱셈-연산)
6. [분산 처리](#분산-처리)
7. [LLM에서의 역할](#llm에서의-역할)

---

## 개요

Feed Forward 블록은 Transformer 레이어의 핵심 구성 요소 중 하나로, Attention 블록 다음에 위치합니다. 각 토큰에 대해 독립적으로 비선형 변환을 수행하여 모델의 표현력을 향상시킵니다.

### 특징

- **위치**: Attention 블록 이후
- **독립성**: 각 토큰 위치에서 독립적으로 계산
- **비선형성**: SiLU 활성화 함수와 곱셈 연산으로 복잡한 패턴 학습
- **차원 확장**: 입력 차원을 확장했다가 다시 축소하는 구조

---

## Feed Forward 블록의 구조

### 전체 흐름

```
입력 (Attention 출력)
    ↓
Residual Connection (MERGE_ADD)
    ↓
RMS Normalization
    ↓
W1 행렬 곱셈 (Up Projection) → D
    ↓
W3 행렬 곱셈 (Up Projection) → L
    ↓
SiLU(D) ⊙ L (비선형 변환)
    ↓
W2 행렬 곱셈 (Down Projection)
    ↓
출력
```

### 코드 구조

```cpp
// distributed-llama/src/llm.cpp:405-554
// 1. Residual Connection
ff.addOp(OP_MERGE_ADD, "block_merge_add2", ...);

// 2. RMS Normalization
ff.addOp(OP_INV_RMS, "block_norm_pre_1", ...);
ff.addOp(OP_RMS_NORM, "block_norm_1", ...);

// 3. W1, W3 행렬 곱셈
ff.addOp(OP_MATMUL, "block_matmul_w1", ...);
ff.addOp(OP_MATMUL, "block_matmul_w3", ...);

// 4. 비선형 변환
ff.addOp(OP_SILU, "block_act", ...);  // SiLU(D)
ff.addOp(OP_MUL, "block_mul", ...);   // SiLU(D) ⊙ L

// 5. W2 행렬 곱셈
ff.addOp(OP_MATMUL, "block_matmul_w2", ...);
```

---

## 수학적 표현

### 전체 수식

```
FF(X) = (SiLU(XW_1) ⊙ (XW_3))W_2
```

### 단계별 분해

1. **Up Projection**:
   ```
   D = XW_1  (shape: [batch, seq_len, dim] × [dim, ffDim] → [batch, seq_len, ffDim])
   L = XW_3  (shape: [batch, seq_len, dim] × [dim, ffDim] → [batch, seq_len, ffDim])
   ```

2. **비선형 변환**:
   ```
   H = SiLU(D) ⊙ L
   ```
   - `SiLU(x) = x · sigmoid(x)`: Swish 활성화 함수
   - `⊙`: 요소별 곱셈 (Hadamard product)

3. **Down Projection**:
   ```
   Y = HW_2  (shape: [batch, seq_len, ffDim] × [ffDim, dim] → [batch, seq_len, dim])
   ```

### 차원 변화

```
입력: [batch, seq_len, dim]
    ↓ (W1, W3)
중간: [batch, seq_len, ffDim]  (확장)
    ↓ (비선형 변환)
중간: [batch, seq_len, ffDim]  (유지)
    ↓ (W2)
출력: [batch, seq_len, dim]    (축소)
```

---

## 구현 상세

### 1. Residual Connection

```cpp
// distributed-llama/src/llm.cpp:407-411
ff.addOp(
    OP_MERGE_ADD, "block_merge_add2", layerIndex,
    pointerBatchConfig(SRC_PIPE, zqPipeIndex),
    pointerBatchConfig(SRC_BUFFER, xBufferIndex),
    size0(),
    NnMergeAddOpCodeConfig{});
```

- **역할**: Attention 블록의 출력을 Feed Forward 입력과 더함
- **효과**: 그래디언트 흐름 개선, 학습 안정성 향상

### 2. RMS Normalization

```cpp
// distributed-llama/src/llm.cpp:413-423
ff.addOp(
    OP_INV_RMS, "block_norm_pre_1", layerIndex,
    pointerBatchConfig(SRC_BUFFER, xBufferIndex),
    pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
    size0(),
    NnInvRmsOpConfig{h->normEpsilon, 1});
ff.addOp(
    OP_RMS_NORM, "block_norm_1", layerIndex,
    pointerBatchConfig(SRC_BUFFER, xBufferIndex),
    pointerBatchConfig(SRC_BUFFER, yBufferIndex),
    n.rmsNormSize,
    NnRmsNormOpConfig{invRmsBufferIndex, 1});
```

- **역할**: 입력을 정규화하여 학습 안정성 향상
- **수식**: `RMSNorm(x) = x / sqrt(mean(x²) + ε)`

### 3. W1, W3 행렬 곱셈 (Up Projection)

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

- **W1**: 첫 번째 Up Projection 행렬
- **W3**: 두 번째 Up Projection 행렬 (SwiGLU 구조)
- **차원**: `[dim, ffDim]` → 출력 `[batch, seq_len, ffDim]`

### 4. 비선형 변환 (SiLU + 곱셈)

```cpp
// distributed-llama/src/llm.cpp:521-532
ff.addOp(
    OP_SILU, "block_act", layerIndex,
    pointerBatchConfig(SRC_BUFFER, dBufferIndex),
    pointerBatchConfig(SRC_BUFFER, dBufferIndex),
    size0(),
    NnSiluOpCodeConfig{});
ff.addOp(
    OP_MUL, "block_mul", layerIndex,
    pointerBatchConfig(SRC_BUFFER, dBufferIndex),
    pointerBatchConfig(SRC_BUFFER, dBufferIndex),
    size0(),
    NnMulOpCodeConfig{lBufferIndex});
```

- **SiLU**: `SiLU(x) = x · sigmoid(x)`
- **곱셈**: `SiLU(D) ⊙ L` (요소별 곱셈)
- **효과**: SwiGLU 활성화 함수로 강력한 비선형성 제공

### 5. W2 행렬 곱셈 (Down Projection)

```cpp
// distributed-llama/src/llm.cpp:541-546
ff.addOp(
    OP_MATMUL, "block_matmul_w2", layerIndex,
    pointerBatchConfig(SRC_BUFFER, dqBufferIndex),
    pointerBatchConfig(SRC_BUFFER, yBufferIndex),
    size2D(h->weightType, n.w2Slice.n0, n.w2Slice.d),
    NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});
```

- **역할**: 확장된 차원을 원래 차원으로 복원
- **차원**: `[ffDim, dim]` → 출력 `[batch, seq_len, dim]`

---

## 행렬 곱셈 연산

### W1, W3 (Up Projection)

#### 수학적 표현

```
D = XW_1  (shape: [batch, seq_len, dim] × [dim, ffDim] → [batch, seq_len, ffDim])
L = XW_3  (shape: [batch, seq_len, dim] × [dim, ffDim] → [batch, seq_len, ffDim])
```

#### 차원 설명

- **입력 X**: `[batch, seq_len, dim]`
  - `batch`: 배치 크기
  - `seq_len`: 시퀀스 길이
  - `dim`: 모델의 기본 차원 (예: 4096)

- **가중치 W1, W3**: `[dim, ffDim]`
  - `dim`: 입력 차원과 일치
  - `ffDim`: Feed Forward 차원 (일반적으로 `dim`의 2-4배)
  - 예: `dim=4096`, `ffDim=11008` (Llama 7B)

- **출력 D, L**: `[batch, seq_len, ffDim]`
  - 차원이 확장되어 더 넓은 표현 공간에서 변환 수행

### W2 (Down Projection)

#### 수학적 표현

```
Y = HW_2  (shape: [batch, seq_len, ffDim] × [ffDim, dim] → [batch, seq_len, dim])
```

여기서 `H = SiLU(D) ⊙ L`입니다.

#### 차원 설명

- **입력 H**: `[batch, seq_len, ffDim]`
- **가중치 W2**: `[ffDim, dim]`
- **출력 Y**: `[batch, seq_len, dim]`
  - 원래 차원으로 복원

---

## 분산 처리

### W1, W3 슬라이싱

```cpp
// distributed-llama/src/llm.cpp:173-175
n.w1Slice = sliceRowMatmul(h->weightType, nNodes, h->dim, ffDim);
n.w3Slice = sliceRowMatmul(h->weightType, nNodes, h->dim, ffDim);
```

- **방식**: Row Matmul (출력 차원 분할)
- **분할**: 각 노드는 `ffDim/nNodes` 크기의 부분 결과 생성
- **통신**: All-Gather로 전체 결과 수집

#### 예시

```
전체: [4096, 11008] (dim=4096, ffDim=11008)
노드 수: 4
각 노드: [4096, 2752] (ffDim/nNodes = 2752)
→ All-Gather 후: [4096, 11008]
```

### W2 슬라이싱

```cpp
// distributed-llama/src/llm.cpp:174
n.w2Slice = sliceColMatmul(h->weightType, nNodes, ffDim, h->dim);
```

- **방식**: Col Matmul (입력 차원 분할)
- **분할**: 각 노드는 `ffDim/nNodes` 크기의 부분 입력 처리
- **통신**: All-Reduce로 부분 결과 합산

#### 예시

```
전체: [11008, 4096] (ffDim=11008, dim=4096)
노드 수: 4
각 노드: [2752, 4096] (ffDim/nNodes = 2752)
→ All-Reduce (합산) 후: [4096] (출력 차원)
```

### 동기화

```cpp
// distributed-llama/src/llm.cpp:554
ff.addSync(zqPipeIndex, SYNC_NODE_SLICES);
```

- **시점**: Feed Forward 블록 완료 후
- **목적**: 노드 간 부분 결과 통합
- **필요성**: 다음 레이어로 전달하기 전 완전한 결과 필요

---

## LLM에서의 역할

### 1. 비선형 변환

- **선형 변환의 한계**: Attention 블록은 주로 선형 변환과 어텐션 메커니즘으로 구성
- **비선형성 추가**: Feed Forward 블록이 SiLU와 곱셈 연산으로 강력한 비선형성 제공
- **복잡한 패턴 학습**: 선형 변환만으로는 불가능한 복잡한 함수 근사

### 2. 표현력 확장

- **차원 확장**: `dim` → `ffDim` (일반적으로 2-4배 확장)
- **넓은 표현 공간**: 더 넓은 차원에서 특징 추출 및 변환
- **정보 처리**: Attention에서 추출한 정보를 더 풍부하게 변환

### 3. 정보 압축

- **차원 축소**: `ffDim` → `dim`으로 복원
- **핵심 정보 추출**: 확장된 공간에서 변환된 정보를 원래 차원으로 압축
- **효율성**: 다음 레이어로 전달하기 적합한 형태로 변환

### 4. Residual Connection과의 협력

- **정보 보존**: Residual Connection으로 원본 정보 보존
- **변환 추가**: Feed Forward로 새로운 정보 추가
- **균형**: 보존과 변환의 균형으로 학습 안정성 향상

### 5. SwiGLU 활성화 함수

- **SwiGLU**: `SiLU(D) ⊙ L` 형태의 활성화 함수
- **장점**: 기존 ReLU나 GELU보다 더 강력한 비선형성
- **효과**: 모델 성능 향상에 기여

---

## Feed Forward 차원 설정

### 일반적인 설정

```cpp
// distributed-llama/src/llm.cpp:154-157
NnUint ffDim = h->hiddenDim;

if (h->archType == QWEN3_MOE)
    ffDim = h->moeHiddenDim;
```

- **일반 모델**: `ffDim = hiddenDim` (예: 4096 → 11008)
- **MoE 모델**: `ffDim = moeHiddenDim` (MoE Expert의 내부 차원)

### 차원 비율

- **Llama**: `ffDim ≈ 2.7 × dim` (예: 4096 → 11008)
- **GPT**: `ffDim = 4 × dim` (예: 768 → 3072)
- **모델에 따라 다름**: 아키텍처에 따라 최적 비율이 다름

---

## 요약

### 핵심 포인트

1. **구조**: Up Projection → 비선형 변환 → Down Projection
2. **차원 변화**: `dim` → `ffDim` → `dim`
3. **비선형성**: SiLU + 곱셈으로 강력한 비선형 변환
4. **분산 처리**: Row Matmul (W1, W3) + Col Matmul (W2)
5. **역할**: Attention의 보완, 비선형 변환, 표현력 확장

### 연산 순서

```
입력 [batch, seq_len, dim]
    ↓ W1, W3
중간 [batch, seq_len, ffDim]
    ↓ SiLU + 곱셈
중간 [batch, seq_len, ffDim]
    ↓ W2
출력 [batch, seq_len, dim]
```

---

## 참고 자료

- 코드 위치: `distributed-llama/src/llm.cpp:405-554`
- 행렬 곱셈 가이드: `MATRIX_MULTIPLICATION_GUIDE.md`
- MoE 가이드: `MOE_GUIDE.md` (MoE 모델의 경우)

---

**작성일**: 2024  
**버전**: 1.0

