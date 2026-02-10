# Communication Bottleneck And Parallel Policies

This document is a code-backed analysis of how communication bottlenecks are mitigated via parallelism policies in:

- distributed-llama: `/home/ubuntu/distributed-llama`
- vLLM: `/home/ubuntu/vllm`
- TensorRT-LLM: `/home/ubuntu/TensorRT-LLM`

Goal: understand (1) what each project chooses as its *parallelism axes* and (2) what each does to reduce or hide communication overhead.

## Terminology (Practical)

- TP (tensor parallel): shard tensors/weights across ranks; needs collectives (all-reduce / all-gather / reduce-scatter).
- PP (pipeline parallel): shard layers across ranks; needs send/recv between stages.
- SP (sequence parallel): shard sequence dimension; often turns some all-reduce into reduce-scatter + all-gather.
- EP (expert parallel / MoE): route tokens to experts; often needs all-to-all-like exchange.
- Collective fusion: fuse collective + compute (e.g., GEMM+reduce-scatter) or fuse all-reduce with pointwise ops.

## vLLM (Local: `/home/ubuntu/vllm`)

### 1) What parallel axes exist

vLLM exposes tensor-model-parallel collectives via the TP group:

- `tensor_model_parallel_all_reduce` -> `get_tp_group().all_reduce(...)` in `/home/ubuntu/vllm/vllm/distributed/communication_op.py:12`
- `tensor_model_parallel_all_gather` -> `get_tp_group().all_gather(...)` in `/home/ubuntu/vllm/vllm/distributed/communication_op.py:17`
- `tensor_model_parallel_reduce_scatter` -> `get_tp_group().reduce_scatter(...)` in `/home/ubuntu/vllm/vllm/distributed/communication_op.py:24`

Those functions are “policy hooks”: higher layers decide where to call all-reduce vs all-gather vs reduce-scatter.

### 2) Communication backends (how TP collectives are executed)

#### GPU path: layered fallback for all-reduce

`CudaCommunicator.all_reduce` implements a policy that tries several fast paths before falling back:

- symmetric memory all-reduce via `torch.ops.vllm.all_reduce_symmetric_with_copy` (gated by heuristics)
- quick all-reduce (ROCm MI300 series) via `QuickAllReduce`
- custom all-reduce (intra-node, NVLink/P2P-dependent) via `CustomAllreduce`
- symmetric-memory communicator path
- PyTorch/NCCL fallback

See `/home/ubuntu/vllm/vllm/distributed/device_communicators/cuda_communicator.py:25` and the selection logic in `/home/ubuntu/vllm/vllm/distributed/device_communicators/cuda_communicator.py:127`.

#### Custom all-reduce is explicitly “intra-node only”

`CustomAllreduce` refuses to initialize when the process group spans multiple nodes:

- “disabled because this process group spans across nodes” in `/home/ubuntu/vllm/vllm/distributed/device_communicators/custom_all_reduce.py:90`

That is a key policy difference vs distributed-llama: vLLM uses very aggressive intra-node techniques when topology allows.

#### CPU path: shared-memory all-gather/allreduce on x86

`CpuCommunicator` can swap torch.distributed for a shared-memory implementation if:

- arch is x86
- shm ops exist (`torch.ops._C.init_shm_manager`)
- group name starts with TP/PP

See `/home/ubuntu/vllm/vllm/distributed/device_communicators/cpu_communicator.py:28`.

Shared-memory all-gather/allreduce entrypoints exist in `/home/ubuntu/vllm/csrc/cpu/shm.cpp:755` (`shm_all_gather`) and `/home/ubuntu/vllm/csrc/cpu/shm.cpp:780` (`shm_allreduce`).

### 3) “Communication bottleneck” mitigations used by vLLM

#### (A) Replace expensive collectives with reduce-scatter/all-gather patterns where possible

Sequence-parallel patterns explicitly transform an all-reduce-centered flow into reduce-scatter + all-gather.

- Helper methods `_reduce_scatter` and `_all_gather` in `/home/ubuntu/vllm/vllm/compilation/sequence_parallelism.py:122` and `/home/ubuntu/vllm/vllm/compilation/sequence_parallelism.py:127`
- Example replacement: “all_reduce + rmsnorm” -> “reduce_scatter + rmsnorm + all_gather” in `/home/ubuntu/vllm/vllm/compilation/sequence_parallelism.py:141`

This is a policy to reduce synchronization overhead and improve overlap potential.

#### (B) Collective + GEMM fusion (reduce launch overhead; increase overlap)

vLLM has compile-time pattern replacements that fuse:

- `mm` + `reduce_scatter` -> `torch.ops.symm_mem.fused_matmul_reduce_scatter` in `/home/ubuntu/vllm/vllm/compilation/collective_fusion.py:59`
- `all_gather` + `mm` -> `torch.ops.symm_mem.fused_all_gather_matmul` in `/home/ubuntu/vllm/vllm/compilation/collective_fusion.py:92`

This is a direct strategy to reduce the “collective kernel launch + compute kernel launch” overhead and to improve effective bandwidth.

#### (C) P2P NCCL send/recv for KV transfer (avoid global collectives)

The KV transfer subsystem includes a P2P NCCL engine that explicitly sets NCCL tuning env vars and uses per-peer communicators.

- Context manager setting NCCL env vars in `/home/ubuntu/vllm/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py:35`
- Creation of a 2-rank NCCL communicator via `ncclCommInitRank(2, ...)` in `/home/ubuntu/vllm/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py:204`

This is a policy choice: for some flows, use targeted P2P rather than a cluster-wide collective.

## TensorRT-LLM (Local: `/home/ubuntu/TensorRT-LLM`)

### 1) Parallel axes and mapping

TensorRT-LLM models expose a mapping of ranks into TP/PP (and optionally CP/EP depending on build).

- Mapping and group creation lives under `tensorrt_llm.mapping` (repo-wide), and the PyTorch flow uses a `Mapping` object heavily.

### 2) All-reduce as a first-class, strategy-driven module

The PyTorch implementation provides workspace allocation and multiple strategies:

- Allocate all-reduce fusion workspace via `CustomAllReduceHelper.allocate_allreduce_fusion_workspace(...)` in `/home/ubuntu/TensorRT-LLM/tensorrt_llm/_torch/distributed/ops.py:21`
- `AllReduce` module lists strategy meanings and supported fusion patterns (e.g., residual+rmsnorm) in `/home/ubuntu/TensorRT-LLM/tensorrt_llm/_torch/distributed/ops.py:455`

The user-facing model config maps string values to `AllReduceStrategy`:

- `get_all_reduce_strategy` mapping in `/home/ubuntu/TensorRT-LLM/tensorrt_llm/_torch/model_config.py:184`

### 3) “Communication bottleneck” mitigations used by TensorRT-LLM

#### (A) Fused all-reduce patterns (reduce bandwidth *and* kernel overhead)

TensorRT-LLM explicitly rewrites graphs to fuse “allreduce + residual add + RMSNorm” into a single call to `torch.ops.trtllm.allreduce` with a fusion op.

- Pattern registration uses `AllReduceFusionOp.RESIDUAL_RMS_NORM` in `/home/ubuntu/TensorRT-LLM/tensorrt_llm/_torch/compilation/patterns/ar_residual_norm.py:19`
- Replacement calls `torch.ops.trtllm.allreduce(..., AllReduceFusionOp.RESIDUAL_RMS_NORM, ...)` in `/home/ubuntu/TensorRT-LLM/tensorrt_llm/_torch/compilation/patterns/ar_residual_norm.py:57`

This is the same “fuse communication with pointwise ops” idea as vLLM, but implemented via TensorRT-LLM’s custom ops.

#### (B) Topology-aware / strategy-aware all-reduce

The `AllReduce` module documents strategy choices (NCCL vs user-buffer vs min-latency vs low-precision) and uses a strategy parameter to pick kernels:

- Strategy description in `/home/ubuntu/TensorRT-LLM/tensorrt_llm/_torch/distributed/ops.py:468`

#### (C) Auto-parallel integration (explicit comm patterns)

The auto-parallel subsystem deals in explicit communication patterns (`all_gather`, `reduce_scatter`, `all_to_all`, `all_reduce`) and can create all-reduce plugins:

- `create_allreduce_plugin` imported in `/home/ubuntu/TensorRT-LLM/tensorrt_llm/auto_parallel/parallelization.py:17`
- `init_all_reduce_helper` imported in `/home/ubuntu/TensorRT-LLM/tensorrt_llm/auto_parallel/parallelization.py:22`

This is relevant to “bottleneck mitigation” because it makes communication part of the optimization/search space.

## distributed-llama (Local: `/home/ubuntu/distributed-llama`)

### 0) Architecture and constraints

distributed-llama uses an explicit ROOT/WORKER split:

- Root loads the model, forwards to workers, and synchronizes network state.
- Workers run their slice.

See `/home/ubuntu/distributed-llama/README.md:66`.

Practical constraints documented/observed:

- multi-node, CPU-first networking (TCP sockets)
- C++11 constraints (no `std::barrier`)
- performance bottleneck is typically network sync

### 1) Compute parallelism policy (what is sharded)

The project’s intended policy is “tensor parallelism by dimension”, not pipeline parallelism:

- “레이어를 나누지 않습니다… 각 레이어의 연산을 차원 기준으로 분할합니다” in `/home/ubuntu/distributed-llama/NODE_DISTRIBUTION_GUIDE.md:9`

Synchronization is applied at layer boundaries:

- “레이어마다 동기화(All‑Gather / All‑Reduce) 가 발생” in `/home/ubuntu/distributed-llama/NODE_DISTRIBUTION_GUIDE.md:14`

In the actual graph construction, synchronization is injected via `addSync(..., SYNC_NODE_SLICES)`:

- after attention output projection in `/home/ubuntu/distributed-llama/src/llm.cpp:403`
- after feed-forward in `/home/ubuntu/distributed-llama/src/llm.cpp:554`
- after final logits in `/home/ubuntu/distributed-llama/src/llm.cpp:599`

### 2) Network topology used by collectives

The networking layer supports a full-mesh socket mapping, but collective algorithms can choose to use only a subset.

Socket mapping helper:

- `getSocketIndexForNode(myNodeIndex, peerNode)` in `/home/ubuntu/distributed-llama/src/nn/nn-network.cpp:802`

Original O(n^2) all-to-all implementation (kept for compatibility):

- `syncNodeSlices_alltoall(...)` in `/home/ubuntu/distributed-llama/src/nn/nn-network.cpp:759`

### 3) Transport-level tactics (chunking, non-blocking mode, monitoring)

Transport chunk size is fixed:

- `#define MAX_CHUNK_SIZE 4096` in `/home/ubuntu/distributed-llama/src/nn/nn-network.cpp:28`

`NnNetwork::write` and `NnNetwork::read` stream the buffer in `MAX_CHUNK_SIZE` pieces:

- write loop in `/home/ubuntu/distributed-llama/src/nn/nn-network.cpp:408`
- read loop in `/home/ubuntu/distributed-llama/src/nn/nn-network.cpp:426`

`writeMany/readMany` use `send/recv` directly and can spin when `EAGAIN` occurs (non-blocking mode):

- `writeMany` uses `send(..., chunkSize, 0)` in `/home/ubuntu/distributed-llama/src/nn/nn-network.cpp:463`
- `readMany` uses `recv(..., io->size, 0)` in `/home/ubuntu/distributed-llama/src/nn/nn-network.cpp:506`

Runtime knobs:

- non-blocking network mode: `NnNetwork::setTurbo(true)` sets sockets non-blocking, see `/home/ubuntu/distributed-llama/src/nn/nn-network.cpp:402` and usage in `/home/ubuntu/distributed-llama/src/app.cpp:283`
- built-in network performance monitoring: enabled in `/home/ubuntu/distributed-llama/src/app.cpp:288` and reported via `/home/ubuntu/distributed-llama/src/nn/nn-network.hpp:105`

### 4) The key collective policy: “Star All-Reduce” for SYNC_NODE_SLICES

#### What is executed today

`syncNodeSlices` currently always selects Star All-Reduce (root-centric gather + reduce + broadcast):

- `syncNodeSlices(...)` calls `syncNodeSlices_starAllReduce(...)` in `/home/ubuntu/distributed-llama/src/nn/nn-network.cpp:1309`

Star All-Reduce implementation:

- `syncNodeSlices_starAllReduce(...)` in `/home/ubuntu/distributed-llama/src/nn/nn-network.cpp:1176`
- Root side reads full buffers from each worker and reduces them with `reduceSum(...)` in `/home/ubuntu/distributed-llama/src/nn/nn-network.cpp:1192`

`reduceSum` is a byte-safe elementwise sum used by the All-Reduce:

- `reduceSum(...)` in `/home/ubuntu/distributed-llama/src/nn/nn-network.cpp:1002`

#### Why this reduces the “communication bottleneck” in practice

The project intentionally converged on all-reduce as the single primitive (instead of mixing all-gather and all-reduce), and then encodes “all-gather-like” behavior by writing each rank’s slice into disjoint offsets of a full-sized buffer.

This design is described in `/home/ubuntu/ALL_REDUCE_ALL_GATHER_CHANGES.md:15` and explained with the key invariant:

- each node only writes non-zero values in its assigned slice region
- other regions are zero/initial
- sum-reduction reconstructs the full tensor because slices do not overlap

See the worked explanation under “All-Reduce 동작” in `/home/ubuntu/ALL_REDUCE_ALL_GATHER_CHANGES.md:61`.

#### Tradeoffs (what becomes the bottleneck)

Star All-Reduce makes the ROOT socket a critical path.

An example benchmark report identifies `Socket 0` as the bottleneck and shows gather dominating the total sync time:

- “Bottleneck: Socket 0” in `/home/ubuntu/PERFORMANCE_BENCHMARK_REPORT.md:61`
- gather vs broadcast ratio in `/home/ubuntu/PERFORMANCE_BENCHMARK_REPORT.md:50`

This matches the star topology expectation: you reduce link fan-out at the cost of concentrating load at the root.

### 5) Alternative collective algorithms exist (currently not selected)

The codebase contains ring-based implementations:

- ring all-gather: `ringAllGather(...)` in `/home/ubuntu/distributed-llama/src/nn/nn-network.cpp:824`
- ring all-reduce (reduce-scatter + all-gather): `syncNodeSlices_ringAllReduce(...)` in `/home/ubuntu/distributed-llama/src/nn/nn-network.cpp:1084`

The design discussion and prior attempts are documented in `/home/ubuntu/distributed-llama/NETWORK_OPTIMIZATION_REPORT.md:114`.

## Quick comparison (Policy-oriented)

| Item | distributed-llama | vLLM | TensorRT-LLM |
|------|------------------|------|--------------|
| Primary target | multi-node CPU cluster | GPU inference + multi-backend | GPU inference (TRT kernels/plugins) |
| Parallel axes | TP-by-dimension (no PP) | TP/PP + SP/EP options | TP/PP (+ optional CP/EP in some flows) |
| Main “bottleneck mitigation” lever | change collective topology (O(n^2)->O(n)), unify sync primitive | fuse collectives; choose fast allreduce backends; SP reduce-scatter/all-gather | fused allreduce patterns; topology-aware allreduce strategy; plugin workspaces |
| Typical collective primitives | star all-reduce over sockets | NCCL all-reduce/all-gather/reduce-scatter (+ custom) | NCCL/user-buffer/min-latency allreduce (+ fusion ops) |
| P2P usage | socket send/recv (full mesh mapping) | NCCL send/recv for KV transfer | PP send/recv + custom comm ops |

## Practical checklist for bottleneck work on distributed-llama

1) Confirm where sync time goes

- enable monitoring (already enabled in root inference path): `/home/ubuntu/distributed-llama/src/app.cpp:288`
- inspect per-op and per-socket stats: `/home/ubuntu/distributed-llama/src/nn/nn-network.hpp:105`

2) Confirm what is being synchronized

- locate `addSync(..., SYNC_NODE_SLICES)` sites: `/home/ubuntu/distributed-llama/src/llm.cpp:403`, `/home/ubuntu/distributed-llama/src/llm.cpp:554`, `/home/ubuntu/distributed-llama/src/llm.cpp:599`
- confirm the sync implementation path: `/home/ubuntu/distributed-llama/src/nn/nn-network.cpp:1325`

3) Confirm the collective algorithm in effect

- `syncNodeSlices(...)` selection: `/home/ubuntu/distributed-llama/src/nn/nn-network.cpp:1309`

4) Know the knobs that can change behavior without code changes

- `--nthreads` (controls CPU compute + some sync fanout like `syncWithRoot` on root): `/home/ubuntu/distributed-llama/README.md:95` and `/home/ubuntu/distributed-llama/src/nn/nn-network.cpp:732`
- `--net-turbo` (non-blocking sockets): `/home/ubuntu/distributed-llama/src/app.cpp:283` and `/home/ubuntu/distributed-llama/src/nn/nn-network.cpp:402`
- `--buffer-float-type` (impacts sync payload size): `/home/ubuntu/distributed-llama/README.md:85`

## Appendix: key source entrypoints

- distributed-llama TP model construction + sync insertion: `/home/ubuntu/distributed-llama/src/llm.cpp`
- distributed-llama synchronization and collectives: `/home/ubuntu/distributed-llama/src/nn/nn-network.cpp`
- vLLM TP collectives API: `/home/ubuntu/vllm/vllm/distributed/communication_op.py`
- vLLM device communicator policy: `/home/ubuntu/vllm/vllm/distributed/device_communicators/cuda_communicator.py`
- vLLM collective fusion: `/home/ubuntu/vllm/vllm/compilation/collective_fusion.py`
- TensorRT-LLM PyTorch allreduce and workspaces: `/home/ubuntu/TensorRT-LLM/tensorrt_llm/_torch/distributed/ops.py`
- TensorRT-LLM compilation fusion patterns: `/home/ubuntu/TensorRT-LLM/tensorrt_llm/_torch/compilation/patterns/ar_residual_norm.py`

## External references (Upstream)

These are useful for cross-checking intent and nomenclature against upstream projects:

- vLLM docs: https://docs.vllm.ai/en/stable/serving/parallelism_scaling/
- vLLM distributed API: https://docs.vllm.ai/en/latest/api/vllm/distributed/
- TensorRT-LLM docs: https://docs.nvidia.com/tensorrt-llm/index.html
- TensorRT-LLM parallel strategy: https://nvidia.github.io/TensorRT-LLM/features/parallel-strategy.html
- TensorRT-LLM overlap scheduler: https://nvidia.github.io/TensorRT-LLM/features/overlap-scheduler.html
