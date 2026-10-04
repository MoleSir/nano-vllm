# Paged Attention CUDA Kernels

本目录下的 kernel 对应 nano-vllm 的 `store_kvcache`（Triton）与 flash-attn 的
`flash_attn_varlen_func` / `flash_attn_with_kvcache`。Rust 侧签名见
[`src/kernels/mod.rs`](./mod.rs)，调用方见 [`src/components/attention.rs`](../components/attention.rs)。

当前全部是 `todo!()`，这份文档用于指导实现。

---

## 0. 背景与约定

### 分页 KV cache 布局

层内 K/V 缓存在显存中是连续的：

```text
k_cache: (num_blocks, block_size, num_kv_heads, head_dim)   # 即扁平 (num_blocks*block_size, num_kv_heads*head_dim)
v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
```

- 每个**物理块**（physical block）存 `block_size` 个 token 的 K/V。
- `num_blocks * block_size` 个 token 槽位，槽位 id `slot ∈ [0, num_blocks*block_size)`。
- 单个 token 的向量长度 `D = num_kv_heads * head_dim`，是连续的。
- 记 `S = num_blocks * block_size * D`，则第 `layer` 层的 K/V base 为：
  ```text
  k_base = kv_cache + layer * 2 * S
  v_base = k_base + S
  ```
  （nanovllm 侧的大 cache 形状 `(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)`，
  各层用 `narrow/squeeze` 取出 `(num_blocks, block_size, num_kv_heads, head_dim)` 视图传入。）

### 逻辑位置 → 物理槽位

一条 sequence 有它自己的**页表** `block_table: (max_blocks,)`，`block_table[lb]` 表示该 seq
第 `lb` 个逻辑块落在哪个物理块。逻辑位置 `p`（从 0 开始）的槽位：

```text
lb   = p / block_size
off  = p % block_size
slot = block_table[lb] * block_size + off
```

### Context

三个 kernel 都从 [`components::Context`](../components/context.rs) 取元数据：

| 字段 | 形状 | 含义 |
|---|---|---|
| `is_prefill` | bool | prefill（变长）还是 decode（每 seq 1 token）|
| `slot_mapping` | `(total_tokens,)` i32 | 本步**新算出**的每个 token 的物理槽位（写入用）|
| `block_tables` | `(num_seqs, max_blocks)` i32 | 页表；prefix cache 时才在 prefill 出现，decode 必有 |
| `context_lens` | `(num_seqs,)` i32 | 每 seq 当前 KV 总长度（读取用）|
| `cu_seqlens_q` | `(num_seqs+1,)` i32 | query 的累积长度（prefill）|
| `cu_seqlens_k` | `(num_seqs+1,)` i32 | key 的累积长度（prefill）|
| `max_seqlen_q/k` | usize | 本批最大 q/k 长度（选 tile / 越界判断用）|

### 通用约束

- dtype：`f32`（luma 目前只有 F32/F64）。
- `head_dim ∈ {32, 64, 128}`（建议），`num_heads % num_kv_heads == 0`（GQA）。
- `block_size` 必须是 **256 的倍数**（nanovllm 固定 256）。
- `scale` 显式传入，通常 `1 / sqrt(head_dim)`。
- 索引统一 `i32`。

---

## 1. `store_kvcache` —— 把新 K/V 写入分页缓存

### 作用

把本步新算出的 K/V（已过 RoPE）按 `slot_mapping` **散布写入**分页缓存。
它是 attention 能读到最新上下文、以及 prefix 能被其他请求复用的前提。
等价于 flash-attn 内部的更新（`reshape_and_cache`）。

### 签名

```rust
pub fn store_kvcache(
    k: &Tensor<Cuda, Float>,          // (num_tokens, num_kv_heads, head_dim)
    v: &Tensor<Cuda, Float>,          // (num_tokens, num_kv_heads, head_dim)
    k_cache: &Tensor<Cuda, Float>,    // (num_blocks, block_size, num_kv_heads, head_dim)
    v_cache: &Tensor<Cuda, Float>,    // 同上
    slot_mapping: &Tensor<Cuda, Int>, // (num_tokens,) i32
) -> anyhow::Result<()>;
```

### 参数含义

| 参数 | 说明 |
|---|---|
| `num_tokens` | 本步要写入的 token 数（prefill 为未缓存的 prompt token 数；decode 为 `num_seqs`，每个 seq 1 个）|
| `k`/`v` | 行 `i` 是第 `i` 个新 token 的 K/V，长度 `D = num_kv_heads * head_dim` |
| `k_cache`/`v_cache` | 该层的缓存；要求 stride(-1)==1 且 stride(1)==D，即可视为 `(num_blocks*block_size, D)` |
| `slot_mapping[i]` | 第 `i` 个 token 的目标槽位 |

### 建议的 kernel / C ABI

```cuda
// grid = (num_tokens), block = (min(1024, D))
__global__ void store_kvcache_kernel(
    const float* k, const float* v,
    float* k_cache, float* v_cache,
    const int* slot_mapping,
    int num_tokens, int D)   // D = kv_heads * head_dim
{
    int t = blockIdx.x;
    long slot = slot_mapping[t];
    const float* ks = k + (long)t * D;
    const float* vs = v + (long)t * D;
    float* kd = k_cache + slot * D;
    float* vd = v_cache + slot * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        kd[i] = ks[i];
        vd[i] = vs[i];
    }
}
```

### 例子

设 `block_size = 4`、`num_kv_heads = 2`、`head_dim = 2`（`D = 4`）、`num_blocks = 3`，
`k_cache` 逻辑上为 `(12, 4)`。

某 seq 有 5 个新 token，`block_table = [1, 0]`：

```text
p=0  lb=0 off=0  slot = 1*4+0 = 4
p=1  lb=0 off=1  slot = 5
p=2  lb=0 off=2  slot = 6
p=3  lb=0 off=3  slot = 7
p=4  lb=1 off=0  slot = 0*4+0 = 0
slot_mapping = [4, 5, 6, 7, 0]
```

写入后：`k_cache[4] = k[0]`，…，`k_cache[0] = k[4]`。
注意：该 token 的 K/V 在本步 forward 里**先写后读**（decode 的 attention 会立刻读它）。

---

## 2. `flash_attn_varlen` —— prefill（变长）attention

### 作用

处理 prefill：把多条变长 query 打包在一起做 flash attention（causal）。
当命中 prefix cache 时，K/V 从**分页缓存**读取（靠 `block_tables`）；否则用刚算出的**连续** K/V。

### 签名

```rust
pub fn flash_attn_varlen(
    q: &Tensor<Cuda, Float>,   // (total_q, num_heads, head_dim)
    k: &Tensor<Cuda, Float>,   // 见下：paged 池 或 连续
    v: &Tensor<Cuda, Float>,
    ctx: &Context,
    scale: f64,
) -> anyhow::Result<Tensor<Cuda, Float>>; // (total_q, num_heads, head_dim)
```

### `k`/`v` 的两种形态

| 情况 | `ctx.block_tables` | `k`/`v` 形状 | 含义 |
|---|---|---|---|
| 无 prefix cache | `None` | `(total_k, num_kv_heads, head_dim)` | 本步新算的连续 K/V，`total_k == total_q` |
| 有 prefix cache | `Some` | `(num_blocks, block_size, num_kv_heads, head_dim)` | 整个分页池；实际读取由页表和长度决定 |

### 使用的 Context 字段

- `ctx.is_prefill == true`
- `ctx.cu_seqlens_q`：`(num_seqs+1,)`，第 `s` 条的 q 区间是 `[cu_q[s], cu_q[s+1])`
- `ctx.cu_seqlens_k`：`(num_seqs+1,)`，第 `s` 条可见的 KV 长度是 `cu_k[s+1] - cu_k[s]`
- `ctx.block_tables`：有 prefix cache 时给出
- `ctx.max_seqlen_q/k`：用于选 tile / 上界

### 语义

对第 `s` 条 sequence：
- query 全局位置 `g = (k_len - q_len) + local_q`（右下对齐）；例如 `k_len=5, q_len=1` → `g=4`。
- K 逻辑位置 `j ∈ [0, k_len)`；causal 可见条件 `j <= g`。
- 有 `block_tables` 时，逻辑位置 `j` 通过 `block_table[ j / block_size ] * block_size + j % block_size` 映射到物理槽位再读取。

### 建议的 kernel（在现有 flash attn 基础上改“K/V 取址”）

```cuda
// grid = (num_seqs, num_heads, num_q_tiles), block = (Br)
// 关键改动：K/V 不再用 j*stride，而是先算 slot
int s = blockIdx.x;
int q_start = cu_seqlens_q[s];
int q_len   = cu_seqlens_q[s+1] - q_start;
int kv_len  = cu_seqlens_k[s+1] - cu_seqlens_k[s];
int q_tile  = blockIdx.z;
int row     = q_tile * Br + threadIdx.x;
if (row < q_len) {
    int g = (kv_len - q_len) + row;          // 本 query 的全局位置
    // 遍历 KV tile...
    for (int jt = 0; jt < ceil_div(kv_len, Bc); ++jt) {
        for (int y = 0; y < Bc; ++y) {
            int j = jt * Bc + y;
            if (j >= kv_len) break;
            const float* kp;
            if (block_table != nullptr) {
                int slot = block_table[s*max_blocks + j / block_size] * block_size + j % block_size;
                kp = k_pool + (long)slot * D;   // D = kv_heads*head_dim
            } else {
                kp = k_contig + ((long)kv_start + j) * D;
            }
            // score = dot(q, kp) * scale（GQA: kv_head = q_head / group）
            // 可见性: j <= g
        }
    }
}
```

### 例子

2 条序列，`block_size=4`，`head_dim=2`，`kv_heads=2`（`D=4`）：

**无 prefix cache**（两条都是全新 prompt）：

```text
seq0: q_len=5, k_len=5
seq1: q_len=3, k_len=3
cu_seqlens_q = [0, 5, 8]
cu_seqlens_k = [0, 5, 8]
block_tables = None
q: (8, num_heads, 2)
k/v: (8, 2, 2)
输出 o: (8, num_heads, 2)
```

**有 prefix cache**（seq0 复用 4 个 token，只算 1 个；seq1 全新 3 个）：

```text
seq0: k_len=5, q_len=1, 复用 block_table=[1,0]
seq1: k_len=3, q_len=3
cu_seqlens_q = [0, 1, 4]
cu_seqlens_k = [0, 5, 8]
context_lens = [5, 3]
block_tables = 2 x max_blocks（pad -1）
q: (4, num_heads, 2)         # total_q = 1 + 3
k/v: (num_blocks, 4, 2, 2)   # 分页池
```

---

## 3. `flash_attn_with_kvcache` —— decode attention

### 作用

decode：每条 sequence 只有 1 个新 query token，从分页缓存里读全部历史 KV 做 attention（causal）。
本步的 K/V 已由 `store_kvcache` 写入缓存，所以这里**只读**。

### 签名

```rust
pub fn flash_attn_with_kvcache(
    q: &Tensor<Cuda, Float>,          // (num_seqs, num_heads, head_dim)
    k_cache: &Tensor<Cuda, Float>,    // (num_blocks, block_size, num_kv_heads, head_dim)
    v_cache: &Tensor<Cuda, Float>,    // 同上
    ctx: &Context,
    scale: f64,
) -> anyhow::Result<Tensor<Cuda, Float>>; // (num_seqs, num_heads, head_dim)
```

> 对应 Python 的 `flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
> cache_seqlens=context_lens, block_table=block_tables, causal=True)`，
> 只是 **不传 k/v**（写入已单独完成），且 `cache_seqlens` = 当前 KV 总长。

### 使用的 Context 字段

- `ctx.is_prefill == false`
- `ctx.context_lens`：`(num_seqs,)`，第 `s` 条的 KV 总长 `L_s`（含当前 token）
- `ctx.block_tables`：`(num_seqs, max_blocks)`
- `ctx.max_seqlen_k`（可选）

### 语义

对第 `s` 条 sequence：query 的全局位置是 `L_s - 1`（每条只有 1 个 query），
可见所有 `j ∈ [0, L_s)` 的 KV，槽位由页表映射：

```text
slot(j) = block_tables[s*max_blocks + j/block_size] * block_size + j%block_size
```

### 建议的 kernel（warp-per-head）

```cuda
// grid = (num_seqs), block = (num_heads * 32)
int s = blockIdx.x;
int L = context_lens[s];
int warp = threadIdx.x / 32;
int lane = threadIdx.x % 32;
int q_head = warp;
int kv_head = q_head / (num_heads / num_kv_heads);
// 每个 lane 持有 head_dim/32 个 q 元素与对应的 o
// 在线 softmax 遍历 j = 0..L：
for (int j = 0; j < L; ++j) {
    int slot = block_tables[s*max_blocks + j/block_size] * block_size + j%block_size;
    const float* kj = k_cache + (long)slot * D + kv_head * head_dim; // D = kv_heads*head_dim
    // partial = dot(q_vec, kj); warp allreduce; score = partial * scale
    // online softmax 更新 m, l, acc（v 同理从 v_cache 取）
}
// 写回 o[s, q_head, :]
```

优化点（可后续再做）：块内 K/V 先 stage 到 shared memory，供共享同一 `kv_head` 的多个 warp 复用；
head_dim 按 lane 切分、warp reduce。

### 例子

`num_seqs=2`、`num_heads=4`、`kv_heads=2`、`head_dim=2`、`block_size=4`：

```text
seq0: context_lens=5, block_table=[1,0]
seq1: context_lens=3, block_table=[2]
q: (2, 4, 2)
k_cache/v_cache: (num_blocks, 4, 2, 2)
输出 o: (2, 4, 2)
```

seq0 的 KV 逻辑位置 → 物理槽位：

```text
j=0..3 → block 1, slot = 4..7
j=4    → block 0, slot = 0
```

seq1：

```text
j=0..2 → block 2, slot = 8,9,10   (假设 block 2 base = 2*4 = 8)
```

---

## 4. 一次 forward 里的调用顺序

以一条请求为例（`Attention::forward`）：

```text
1. q/k/v 投影 + reshape 到 (num_tokens, heads, head_dim)
2. k/v 过 RoPE（按 positions）
3. store_kvcache(k, v, k_cache_layer, v_cache_layer, ctx.slot_mapping)   # 写入
4. if prefill:
       flash_attn_varlen(q, k_or_pool, v_or_pool, ctx, scale)             # 读 + 算
   else:
       flash_attn_with_kvcache(q, k_cache_layer, v_cache_layer, ctx, scale)
5. o reshape 回 (num_tokens, num_heads*head_dim) → o_proj
```

对应 nanovllm 的模型层数：`store_kvcache` 和 attention 调用都是**每层一次**，
`k_cache_layer`/`v_cache_layer` 是该层视图。

---

## 5. 与 Python 版本的对应关系

| Rust (`kernels::*`) | nano-vllm | flash-attn |
|---|---|---|
| `store_kvcache` | `layers/attention.py::store_kvcache` (Triton) | 内部 `update_kv_cache` / `reshape_and_cache` |
| `flash_attn_varlen` | `flash_attn_varlen_func(..., block_table=...)` | `mha_varlen_fwd` |
| `flash_attn_with_kvcache` | `flash_attn_with_kvcache(q.unsqueeze(1), ...)` | `mha_fwd_kvcache` |

差异点：
- 我们用 `f32`（flash-attn 只支持 `fp16/bf16`）。
- `block_size` 固定 256 的倍数。
- decode 的 `positions` 按本地 nano-vllm 用 `len(seq)`（上游 `main` 用 `len-1`）。
- prefill 的 `k/v` 在有无 prefix cache 时择一传入，和 nano-vllm 的
  `if context.block_tables is not None: k, v = k_cache, v_cache` 一致。
