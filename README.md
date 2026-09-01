# AI/机器学习工程 / AI/ML Engineering

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open Source](https://img.shields.io/badge/status-open--source-brightgreen.svg)](#开源状态--open-source-status)
[![Codex Skill](https://img.shields.io/badge/Codex-Skill-black.svg)](SKILL.md)

> 覆盖数据体检、泄漏防护、经典机器学习、深度学习、时序验证、模型解释和 GPU 加速的工程化 Skill。
>
> An engineering-focused skill for data quality, leakage prevention, classical ML, deep learning, time-series validation, explainability, and GPU acceleration.

## 中文介绍

### 能力范围

- 用 EDA 与数据契约定位质量问题
- 防止特征、标签与时间序列泄漏
- 组织 scikit-learn、PyTorch Lightning 与强化学习实验
- 通过 SHAP 等方法解释模型
- 先 profile，再使用 CuPy、Numba 或 CUDA 优化瓶颈

### 适用场景

模型训练、时序预测、特征工程、实验复现、GPU 调优和模型审计。

### 设计方式

`SKILL.md` 是唯一入口，先完成场景分诊，再按需读取 `references/` 中的专题资料。这样既保留关键约束，又避免把所有领域知识一次性装入上下文。脚本仅用于可重复、可验证的机械操作。

### 安装

```powershell
git clone https://github.com/GongGongPlus/skill-ai-ml-engineering.git "$env:USERPROFILE\.codex\skills\ai-ml-engineering"
```

安装后重新打开 Codex 会话，让 Skill 目录重新被发现。也可以直接在任务中点名 `$ai-ml-engineering`。

### 使用示例

```text
请使用 $ai-ml-engineering 处理这个任务，并把已验证事实、工程推论和待验证项分开报告。
```

### 证据与边界

不把单次指标提升当作泛化证据；数据版本、随机种子、代码提交与评估切分必须可追溯。

本仓库提供工作流与判断框架，不替代官方规则、专业认证、生产环境审批或真实设备验证。执行涉及资金、硬件熔丝、生产部署、外部提交等高风险操作前，必须取得明确授权。

## English

### Scope

- Diagnose data quality with EDA and explicit contracts
- Prevent feature, label, and time-series leakage
- Structure scikit-learn, PyTorch Lightning, and reinforcement-learning experiments
- Explain model behavior with SHAP and related methods
- Profile first, then optimize bottlenecks with CuPy, Numba, or CUDA

### Best fit

Model training, forecasting, feature engineering, reproducible experiments, GPU tuning, and model audits.

### Design

`SKILL.md` is the single entry point. It triages the request first and loads only the relevant files under `references/`. This preserves important constraints without loading the entire knowledge base into context. Scripts are reserved for repeatable, verifiable mechanical work.

### Installation

```bash
git clone https://github.com/GongGongPlus/skill-ai-ml-engineering.git "$HOME/.codex/skills/ai-ml-engineering"
```

Start a fresh Codex session after installation so the skill directory is rediscovered. You can also invoke it explicitly as `$ai-ml-engineering`.

### Example prompt

```text
Use $ai-ml-engineering for this task. Separate verified facts, engineering inferences, and items that still need validation.
```

### Evidence boundary

A one-off metric gain is not evidence of generalization; data versions, seeds, code revisions, and evaluation splits must remain traceable.

This repository provides workflows and decision support. It does not replace official rules, professional certification, production approval, or real-device validation. Explicit authorization is required before high-risk actions involving funds, hardware fuses, production deployment, or external submission.

## Repository structure / 仓库结构

```text
skill-ai-ml-engineering/
|-- references/
|   |-- classical-ml.md
|   |-- deep-learning.md
|   |-- eda-data-quality.md
|   |-- gpu-acceleration.md
|   |-- timeseries-leakage.md
|-- SKILL.md
```

Reference topics / 专题资料：

- `references/classical-ml.md`
- `references/deep-learning.md`
- `references/eda-data-quality.md`
- `references/gpu-acceleration.md`
- `references/timeseries-leakage.md`

## Author / 作者

- Author and maintainer / 作者与维护者：**中北大学机器人协会续晋全**
- GitHub: [@GongGongPlus](https://github.com/GongGongPlus)
- Collection index / 总索引：[GongGongPlus/agent-skills-index](https://github.com/GongGongPlus/agent-skills-index)

The author statement identifies the maintainer and original integrator of this repository. Referenced third-party projects retain their own authorship and rights.

作者信息表示本仓库的维护者与原创整合者；被引用的第三方项目仍保留其原作者身份与相关权利。

## 开源状态 / Open-source status

- Visibility / 可见性：**Public / 公开**
- Status / 状态：**Open source / 开源**
- License / 许可证：[MIT](LICENSE)
- Warranty / 担保：按“原样”提供，不承诺适用于特定目的 / Provided as-is, without warranty
- Third-party boundary / 第三方边界：见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

欢迎在遵守许可证与证据边界的前提下使用、修改和分发。Issues 与 Pull Requests 可用于报告可复现问题或提交改进。

Use, modification, and redistribution are welcome under the license and evidence boundaries. Issues and pull requests may be used for reproducible bug reports and improvements.
