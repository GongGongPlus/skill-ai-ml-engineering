# 经典 ML：scikit-learn 与 SHAP

**第一步该做什么**：搭最小 Pipeline 跑通基线（DummyClassifier/Regressor 先行），再谈调参与解释。

## Pipeline 与防泄漏（已确认）

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV

num_cols = ["age", "income"]; cat_cols = ["city"]
pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
pipe = Pipeline([("pre", pre), ("clf", LogisticRegression())])
# 调参 unbiased 估计 = 嵌套 CV
grid = GridSearchCV(pipe, param_grid, cv=5)
scores = cross_val_score(grid, X, y, cv=5)   # 外层再包一层
```

- 预处理**必须**在每折内重拟合；对全量数据 fit_transform 后再切分 = 泄漏。
- 需要缩放的算法：SVM / KNN / NN / PCA / 带正则线性 / K-Means；树模型和朴素贝叶斯不需要。
- 大数据放不下内存：SGDClassifier / MiniBatchKMeans / HistGradientBoosting。

## 不平衡数据

顺序：先 `class_weight="balanced"` → 不够再 imbalanced-learn 重采样（**放训练折内**，Pipeline 里做）→ 评估用 PR-AUC / F1，不看 accuracy。

## SHAP Explainer 决策表

| 模型 | Explainer | 备注 |
|---|---|---|
| 树模型 | TreeExplainer | 最快，exact |
| 线性 | LinearExplainer | — |
| 特征少（<~20） | ExactExplainer | — |
| 通用黑盒 | PermutationExplainer | `max_evals=2*n_features+1` |
| 文本/图像 | PartitionExplainer + masker | — |

关键用法：
- 概率空间解释必须显式：`TreeExplainer(model, data=bg, feature_perturbation="interventional", model_output="probability")`。
- 多分类取类：`explanation[..., class_index]`（新 API），不要用旧 `values[class_index]`，不要跨类平均带符号归因。
- `approximate=True` 传给调用，不是构造器。

## additivity 失败诊断顺序（shap/troubleshooting）

`base_values + values.sum(axis=1)` ≠ 模型输出时，按序查：行列对齐 → 列序/dtype/稀疏性匹配 → 是否同一个已拟合模型 → 输出 index/单位 → 单行复算 → 换 background。**`check_additivity=False` 仅在量化误差并记录后允许**——直接关掉校验等于没解释。

## 如何证明结论可信

① 指标同时报"vs 基线"与"vs 随机猜测"；② 嵌套 CV 分数与单层 CV 分数差距大 → 调参过拟合，信嵌套那个；③ SHAP 结论换 background/样本子集重算一次方向不变，才算稳健。
