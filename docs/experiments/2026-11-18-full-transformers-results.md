# 2026-11-18 CoopES Transformers 完整实验结果

## 运行范围

- 分支：`dev`
- 数据集：OpenBookQA `additional`、ARC-Challenge
- 模型：`/home/model/Qwen2.5-7B-Instruct`
  （`local-d3b8f5a961945310ad260269142108435f9771b72233343424f927bb39408da4`）、
  `/home/model/Meta-Llama-3-8B`
  （`local-c7e012e0b370ab7396b04df1a31fdc8e45380a7fe5d31860f28da7f528137f6d`）
- 数据版本：OpenBookQA `388097ea...`，ARC-Challenge `210d026f...`
- 提示模板 digest：`05f1a1a4160030577241e33b6b9a94c8f97ac6856967fe13f59b63a25afaf76f`
- 数据规模：50 个候选、50 道校准题、500 道测试题
- 选例参数：`k=4`、`M=200`、`epsilon=0.05`
- 后端：Transformers，bfloat16，CUDA `cuda:0`，`max_new_tokens=1`
- 硬件：NVIDIA RTX 4090 24 GB，16 CPU，62 GiB 内存
- 软件：Python 3.12.11、torch 2.13.0+cu132、transformers 5.23.0
- 总耗时：约 2 小时 53 分钟；累计提交 264,500 个模型输入

ARC-Challenge × Qwen 使用 `batch_size=32`；其余三组使用
`batch_size=16`。Qwen 高负载时出现 CUDA allocator 扩容警告，但
所有运行均正常完成，无重试，也无运行错误。

## 汇总结果

Accuracy、Weighted-F1 与解析失败率均为小数。耗时和推理开销来自原始整组运行记录。

### OpenBookQA × Qwen2.5-7B-Instruct

| 方法 | Accuracy | Weighted-F1 | Parse failure |
| --- | ---: | ---: | ---: |
| Zero-shot | 0.9320 | 0.9338 | 0.0040 |
| CoopES Uniform | 0.9300 | 0.9384 | 0.0180 |
| CoopES Weighted | 0.9300 | 0.9384 | 0.0180 |

运行耗时 36.5 分钟；backend calls 5168；submitted inputs 65900；cache hits 500。

### OpenBookQA × Meta-Llama-3-8B

| 方法 | Accuracy | Weighted-F1 | Parse failure |
| --- | ---: | ---: | ---: |
| Zero-shot | 0.0000 | 0.0000 | 1.0000 |
| CoopES Uniform | 0.8180 | 0.8204 | 0.0020 |
| CoopES Weighted | 0.8160 | 0.8181 | 0.0000 |

运行耗时 40.1 分钟；backend calls 5188；submitted inputs 66250；cache hits 0。

### ARC-Challenge × Qwen2.5-7B-Instruct

| 方法 | Accuracy | Weighted-F1 | Parse failure |
| --- | ---: | ---: | ---: |
| Zero-shot | 0.8940 | 0.8946 | 0.0000 |
| CoopES Uniform | 0.9140 | 0.9175 | 0.0060 |
| CoopES Weighted | 0.9080 | 0.9106 | 0.0040 |

运行耗时 46.4 分钟；backend calls 2606；submitted inputs 66550；cache hits 0。

### ARC-Challenge × Meta-Llama-3-8B

| 方法 | Accuracy | Weighted-F1 | Parse failure |
| --- | ---: | ---: | ---: |
| Zero-shot | 0.0000 | 0.0000 | 1.0000 |
| CoopES Uniform | 0.7020 | 0.6973 | 0.0000 |
| CoopES Weighted | 0.7000 | 0.6970 | 0.0000 |

运行耗时 49.7 分钟；backend calls 5152；submitted inputs 65800；cache hits 0。

## 冻结选集

### OpenBookQA × Qwen2.5-7B-Instruct

- CoopES Uniform：`openbookqa:10-1143`, `openbookqa:14-1273`, `openbookqa:84`, `openbookqa:14-245`
- CoopES Weighted：`openbookqa:14-1273`, `openbookqa:84`, `openbookqa:10-1143`, `openbookqa:14-245`

### OpenBookQA × Meta-Llama-3-8B

- CoopES Uniform：`openbookqa:14-6`, `openbookqa:10-131`, `openbookqa:10-626`, `openbookqa:9-864`
- CoopES Weighted：`openbookqa:13-743`, `openbookqa:1466`, `openbookqa:9-864`, `openbookqa:10-566`

### ARC-Challenge × Qwen2.5-7B-Instruct

- CoopES Uniform：`arc_challenge:VASoL_2007_5_39`, `arc_challenge:AKDE&ED_2012_4_33`, `arc_challenge:FCAT_2012_8_7`, `arc_challenge:Mercury_7099365`
- CoopES Weighted：`arc_challenge:FCAT_2012_8_7`, `arc_challenge:VASoL_2007_5_39`, `arc_challenge:Mercury_400639`, `arc_challenge:AIMS_2009_4_8`

### ARC-Challenge × Meta-Llama-3-8B

- CoopES Uniform：`arc_challenge:VASoL_2007_5_39`, `arc_challenge:Mercury_SC_405455`, `arc_challenge:MSA_2013_5_33`, `arc_challenge:Mercury_SC_407691`
- CoopES Weighted：`arc_challenge:MCAS_2004_8_17`, `arc_challenge:Mercury_7222373`, `arc_challenge:Mercury_7166618`, `arc_challenge:VASoL_2007_5_39`

## 结论与限制

- Qwen 的均匀 CoopES 在 ARC-Challenge 上比 Zero-shot 高 2.00 个准确率百分点，在 OpenBookQA 上低 0.20 个百分点。
- Llama 3 基座模型在 ARC-Challenge 和 OpenBookQA 上的均匀 CoopES 准确率分别为 70.20% 和 81.80%，两组均略高于加权版。
- 本地 Llama 权重是基座模型 `Meta-Llama-3-8B`，不是 Instruct 版本。
  分词器没有内置聊天模板，本实验使用 Llama 3 格式模板回退；
  Zero-shot 输出不符合严格单标签解析规则，因此解析失败率为 1.0，
  该组指标不应与论文中的 Llama3-8B 结果直接比较。
- `max_new_tokens=1` 用于执行“只输出标签”的提示约束；Qwen 的解析
  失败率保持在 0.0%–1.8%，Llama 的 4-shot 方法为 0.0%–0.2%。

## 产物位置

- 汇总表：`docs/experiments/2026-11-18-full-transformers-summary.csv`
- Qwen ARC：`runs/full-arc_challenge-qwen/`
- Qwen OpenBookQA：`runs/full-openbookqa-qwen/`
- Llama 3 ARC：`runs/full-arc_challenge-llama3/`
- Llama 3 OpenBookQA：`runs/full-openbookqa-llama3/`
- 运行日志：`runs/full_experiments.log`

每个运行目录均包含 `summary.json`、`metrics.csv`、
`predictions.jsonl`、`candidate_scores.csv`、`selections.json`、
`splits.json`、`inference_metadata.json` 和 `cache.sqlite3`。
