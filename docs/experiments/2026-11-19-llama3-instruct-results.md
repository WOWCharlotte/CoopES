# 2026-11-19 Llama 3 Instruct 完整实验结果

## 实验范围

- 模型：ModelScope `LLM-Research/Meta-Llama-3-8B-Instruct`
- 本地目录：`/home/model/Meta-Llama-3-8B`
- 权重指纹：`local-8bf148d7929b2d6529bb795b3c38856ca49a321ce394769821656133bb67e4ba`
- 后端：Transformers，bfloat16，CUDA `cuda:0`
- 数据规模：50 候选、50 校准题、500 测试题
- 选例参数：`k=4`、`M=200`、`epsilon=0.05`
- 生成设置：`max_new_tokens=1`、`context_window=8192`、`batch_size=16`
- 软件：Python 3.12.11、torch 2.13.0+cu132、transformers 5.23.0

旧的 Llama 3 基座模型实验目录和诊断文件已删除。以下结果全部来自指令
微调后的 Instruct 权重。

## 结果汇总

Accuracy、Weighted-F1 和解析失败率均为小数。耗时和推理开销来自原始整组运行记录。

### ARC-Challenge

| 方法 | Accuracy | Weighted-F1 | Parse failure |
| --- | ---: | ---: | ---: |
| Zero-shot | 0.8140 | 0.8145 | 0.0000 |
| CoopES Uniform | 0.8260 | 0.8255 | 0.0000 |
| CoopES Weighted | 0.8260 | 0.8254 | 0.0000 |

运行耗时 50.6 分钟；backend calls 5208；submitted inputs 66500；retry attempts 0。

### OpenBookQA

| 方法 | Accuracy | Weighted-F1 | Parse failure |
| --- | ---: | ---: | ---: |
| Zero-shot | 0.8720 | 0.8718 | 0.0000 |
| CoopES Uniform | 0.8880 | 0.8882 | 0.0000 |
| CoopES Weighted | 0.8900 | 0.8898 | 0.0000 |

运行耗时 40.8 分钟；backend calls 5212；submitted inputs 66550；retry attempts 0。

## 与旧基座模型运行的主要变化

| 数据集 | 方法 | 基座模型 | Instruct | 变化 |
| --- | --- | ---: | ---: | ---: |
| ARC-Challenge | Zero-shot | 0.000 | 0.814 | +81.40 pp |
| ARC-Challenge | CoopES Uniform | 0.702 | 0.826 | +12.40 pp |
| ARC-Challenge | CoopES Weighted | 0.700 | 0.826 | +12.60 pp |
| OpenBookQA | Zero-shot | 0.000 | 0.872 | +87.20 pp |
| OpenBookQA | CoopES Uniform | 0.818 | 0.888 | +7.00 pp |
| OpenBookQA | CoopES Weighted | 0.816 | 0.890 | +7.40 pp |

本报告三种方法在两个数据集上的解析失败率均为 0。

## 分析

- ARC-Challenge 上两种 CoopES 的准确率均为 82.6%，比 Zero-shot 高 1.20 个百分点。
- OpenBookQA 上加权与均匀 CoopES 的准确率分别为 89.0% 和 88.8%，比 Zero-shot 高 1.80 和 1.60 个百分点。
- ARC-Challenge 的 Instruct CoopES 结果 82.6% 与论文参考值 82.0% 接近。
- 运行中出现过一次 CUDA allocator 扩容警告，但没有导致运行失败或重试。

## 产物

- ARC-Challenge：`runs/full-arc_challenge-llama3/`
- OpenBookQA：`runs/full-openbookqa-llama3/`
- 运行日志：`runs/llama3-instruct-full.log`
- 权重验证：`runs/llama3-instruct-weight-validation.json`
- 汇总 CSV：`docs/experiments/2026-11-19-llama3-instruct-summary.csv`

每个运行目录包含 `summary.json`、`metrics.csv`、`predictions.jsonl`、
`candidate_scores.csv`、`selections.json`、`splits.json`、
`inference_metadata.json` 和 `cache.sqlite3`。
