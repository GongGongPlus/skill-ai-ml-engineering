# GPU 加速与 CUDA Kernel 调优

**第一步该做什么**：判断这份数据/这段代码是否值得上 GPU（数据 ≥10K 元素、高数据并行、算术强度高、显存放得下——四条全满足才继续），然后选加速入口。

## CPU→GPU 库映射与零改码加速（已确认）

| CPU 库 | GPU 库 | 零改码入口 |
|---|---|---|
| NumPy | CuPy | 换 import 或 `cp.get_array_module(x)` 双兼容 |
| pandas | cuDF | `python -m cudf.pandas xxx.py` |
| scikit-learn | cuML | `python -m cuml.accel xxx.py`（字符串 dtype 回退 CPU，需先 LabelEncoder） |
| NetworkX | cuGraph | `NX_CUGRAPH_AUTOCONFIG=True python xxx.py` |
| Faiss | cuVS | API 替换 |
| scikit-image | cuCIM | API 替换 |

安装：`uv add cudf-cu12 / cuml-cu12`（按 CUDA 版本选 `-cu12/-cu13`）；cuGraph/cuSpatial 需要 `--extra-index-url=https://pypi.nvidia.com`。cuSpatial 已归档（冻结 25.04，会锁死 cudf 版本），cuxfilter 已 sunset——新项目不要用。

## 性能常量（工程推论，用于期望管理）

- PCIe 传输 ~12GB/s vs 显存带宽 ~900GB/s：**来回搬运可能吃掉全部加速收益**。
- kernel launch 开销 ~5–20μs：小任务循环 launch 反而更慢。
- 消费级卡 float64 吞吐 = float32 的 1/32。

## CuPy 关键坑

- 计时禁用 `time.perf_counter()`（只测到入队），必须用 `cupyx.profiler.benchmark()`。
- `nvidia-smi` 显存高企可能是 memory pool 缓存：`cp.get_default_memory_pool().free_all_blocks()`。
- 传输语义：`cp.asarray` 已在当前设备则零拷贝，`cp.array` 恒拷贝。

## Numba CUDA 铁律

```python
@cuda.jit
def kernel(a, b, out):
    i = cuda.grid(1)
    if i < a.size:          # MUST：越界守卫，没有它=未定义行为
        out[i] = a[i] + b[i]  # 无返回值，只写输出数组
```
`cuda.syncthreads()` 不能出现在发散分支里；共享内存 bank conflict 用 padding 消除。

## CUDA Kernel 调优优先级（cuda-kernel-optimizer，按收益排序）

1. **P1 Tensor Core**：GEMM/conv 的 TC 利用率 <20% 时启用，收益 8–16×。
2. **P2 混合精度**：FP32 pipe 高 + TC 低 + 精度宽容 → FP16/BF16（累加器保持 FP32）。
3. **P2 coalesced access**：L1 hit <30% 时检查全局访存模式。
4. **P8 bank conflict**：`bank_conflict > 0` 且无 TMA → padding/重排。
5. Roofline 三轴（compute/memory/latency）Δ 全 <0.15 → near_peak，**早停**。

ncu 快路径：`ncu --set full ./app`；TC% 低+GEMM→P1；dram 写入 ≫ 输出→epilogue fusion。

## cuDF 与 pandas 行为差异（迁移必查）

groupby/join 结果默认乱序（要 `sort=True`）；缺失值是 `<NA>` 非 NaN；不支持 `for val in series` 迭代；object dtype 只存字符串。

## 如何证明优化真实有效

① 同命令同条件下 benchmark，提升必须超过运行间方差（±5% 内的 3% 不算）；② 结果数值对齐（容差内）；③ kernel 优化需 SASS 层验证生效（反汇编确认指令真的换了），否则归因可能是超参波动——归因近零说明提速来自别处。
