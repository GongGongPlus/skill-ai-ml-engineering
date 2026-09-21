# 深度学习训练：Lightning / 强化学习 / TimesFM

**第一步该做什么**：确认训练规模与硬件匹配（<500M 参数 → DDP；≥500M → FSDP），再选框架入口。

## PyTorch Lightning 要点

- `training_step` 必须返回 loss；多卡指标日志加 `sync_dist=True`（否则每卡各记各的）。
- 不要手动 `.cuda()`——batch 已在正确设备；不要自己建 DistributedSampler——Lightning 管。
- OOM 处理顺序：gradient checkpointing → 减 batch → FSDP + cpu_offload → accumulate_grad_batches。
- NCCL timeout：先查两卡间 NCCL 版本一致性，再降 `timeout` 前确认不是死锁。

## Stable-Baselines3（强化学习）

- 训练前**必跑** `check_env(env, warn=True)`——多数自定义环境 bug（obs/action 空间不符、dtype 错）它直接抓出来。
- `total_timesteps` 是下界不是目标；学习曲线看 `ep_rew_mean` 平台期。
- replay buffer 不随 `model.save()` 保存；off-policy 多环境时 `gradient_steps=-1`。
- 图像 obs 用 uint8 [0,255]（内部自动 /255），不要自己归一化。
- 算法选择：连续动作 SAC/TD3，离散 DQN，通用起步 PPO；稀疏奖励配 HER。

## TimesFM（零样本时序预测）

```python
model = TimesFm(hparams=..., checkpoint=...)
model.compile(ForecastConfig(...))   # MUST：不 compile 直接 forecast 会 RuntimeError
point, quantile = model.forecast([series1, series2, ...])  # 输入是 list of 1-D arrays，不是 2D 矩阵
```
- `normalize_inputs=True` 与 `fix_quantile_crossing=True` 必开。
- batch size 对照（工程推论）：8GB VRAM→64、16GB→128、24GB→256。
- `infer_is_positive=True` 只用于非负序列；quantile 区间可做异常检测（出界即异常）。
- 使用前先跑 `check_system.py` 验证环境。

## 目标检测调优提分路线（竞赛实测，来源 aic 目标检测 2026-09，已确认）

提分阶梯，每步都要有验证集 mAP 对照，一次只动一个变量：

1. **分辨率阶梯**：训练/推理分辨率按硬件余量逐级上探（实测 1088×1920 → 1280×2272 可继续提分），显存是唯一硬约束；推理用与训练一致的尺度。
2. **多尺度测试时增强（TTA）**：6 尺度 + 水平翻转 + **WBF 融合**（不是 NMS）可稳定提 mAP@50-95；WBF 对低置信度框的处理优于 NMS。
3. **数据加载优化**：prefetch/workers 拉满，GPU 利用率 ≥90% 是前提，否则先解决 IO 瓶颈再谈别的。
4. **类别不平衡定向增强**：先按类别分桶看 AP 与目标尺寸——对低 AP 小目标类（实测 bicycle 中位 69px、boat 137px）做 2-3 倍过采样/针对性增广，比全局增广提分快。
5. **类别平衡微调（class-balanced fine-tune）**：从已收敛模型继续，只对低 AP 类别过采样训练；保留基础类别权重，防止灾难性遗忘。

工程纪律：
- 提交必须用验证集最优权重 + TTA 全流程生成，保证兜底提交存在
- 训练后台跑时用监控轮询，权重落盘后自动核验再进候选池
- 候选权重复用模型汤（model soup）合并可再提一档（实测 E1+C 汤作为微调起点）

## 如何证明训练配置正确

① 单卡 100 step 冒烟跑通过（loss 下降、无 NaN）再上多卡；② 多卡 loss 曲线与单卡对齐（sync_dist 正确性）；③ RL 训练换 3 个 seed 方向一致，单 seed 结论一律"待验证"；④ 预测结果做了 quantile 区间 sanity check（区间不塌缩、不倒挂）。

## ML 工程契约（全网顶级 skill 深挖，来源 ml-engineering-agent-manifesto）

- **项目结构**：src-layout；`docs/design/` 先写设计文档再实现；`data/{raw,interim,processed}` 三段式；数据集禁止入 git（用 DVC/ClearML Data 版本化）。
- **代码门禁**：ruff lint+format（行宽 100）；Google 风格 docstring；mypy `disallow_untyped_defs=true`，public 函数/方法必须类型标注。
- **Definition of Done**：lint 过 + 类型检查过 + 测试过 + 重大行为变更必须更新 design doc，四者缺一不算完成。
- **可观测性**：每次训练开始初始化实验跟踪 Task；至少记录 dataset version、model version、git commit hash、key metrics；禁止只 print 到 stdout 而不写 tracker。