# CoopES

CoopES 的可运行复现实现：在候选示例池上用校准集估计前缀边际贡献，选出
Top-k 示例后冻结选集，并在独立测试集上评价。第一版支持 OpenBookQA /
ARC-Challenge、Ollama / Transformers 后端、随机与单示例基线、均匀与加权
CoopES，以及 SHAP 标准 Shapley 值核对。

## 安装

```bash
python -m pip install -e '.[dev]'
# 按需安装后端或数据依赖
python -m pip install -e '.[data,ollama]'
# 或
python -m pip install -e '.[data,transformers]'
# SHAP 核对
python -m pip install -e '.[shap]'
```

## 运行

先确认 Ollama 已启动且模型可用，然后运行小规模试跑：

```bash
coopes --config configs/smoke.toml --run-dir runs/smoke
```

主实验配置位于 `configs/main.toml`。每次运行会在目标目录写入：

- `config.json`、`splits.json`、`inference_metadata.json`
- `candidate_scores.csv`、`selections.json`
- `predictions.jsonl`、`metrics.csv`、`summary.json`
- `cache.sqlite3`，用于断点续跑和复用逐题生成结果

`configs/smoke-transformers.toml` 展示本地 Transformers 后端配置。正式
实验前应把 `model_digest` 固定为 `ollama show` 返回的摘要，或为
Transformers 配置固定模型提交。

## 测试

```bash
pytest
ruff check src tests
```

测试覆盖数据标准化、提示顺序与答案解析、采样计数、排名、SHAP、推理
缓存与重试、评分指标、运行产物和端到端编排。
