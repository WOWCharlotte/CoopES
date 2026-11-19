# CoopES 运行结果文件说明


```text
configs/full-arc_challenge-qwen.toml
runs/full-arc_challenge-qwen/
```

运行入口为：

```bash
coopes \
  --config configs/full-arc_challenge-qwen.toml \
  --run-dir runs/full-arc_challenge-qwen
```

## 总体流程

`src/coopes/cli.py:542` 的 `run_experiment()` 依次完成配置加载、数据切分、候选评分、冻结选例和测试集评估。各文件的生产关系如下：

```text
config.toml
   │
   ├─ load_config() ───────────────→ config.json
   │
   ├─ load_splits() ───────────────→ splits.json
   │
   ├─ open_engine()
   │      ├────────────────────────→ cache.sqlite3
   │      └────────────────────────→ inference_metadata.json
   │
   ├─ make_utility()
   │  build_selection_plan()
   │      ├────────────────────────→ candidate_scores.csv
   │      └────────────────────────→ selections.json
   │
   ├─ evaluate_trial()
   │      └────────────────────────→ predictions.jsonl
   │
   └─ write_evaluation_artifacts()
          ├────────────────────────→ metrics.csv
          └────────────────────────→ summary.json
```

一次完整运行的典型目录结构：

```text
runs/full-arc_challenge-qwen/
├── cache.sqlite3
├── candidate_scores.csv
├── config.json
├── inference_metadata.json
├── metrics.csv
├── predictions.jsonl
├── selections.json
├── splits.json
└── summary.json
```


## 文件概览

| 文件 | 主要生产者 | 内容 |
| --- | --- | --- |
| `config.json` | `_config_payload()` | 实验配置快照 |
| `splits.json` | `_splits_payload()` | 候选池、标定集和测试集快照 |
| `cache.sqlite3` | `InferenceEngine.generate_batch()` | 模型生成结果持久缓存 |
| `inference_metadata.json` | `open_engine()` | 后端、模型版本和运行时环境 |
| `candidate_scores.csv` | `write_selection_artifacts()` | 候选示例的效用或边际贡献分数 |
| `selections.json` | `build_selection_plan()` | 所有方法冻结后的 few-shot 选集 |
| `predictions.jsonl` | `write_evaluation_artifacts()` | 每个 trial 的逐题预测和判定 |
| `metrics.csv` | `write_evaluation_artifacts()` | 每个 trial 的聚合指标 |
| `summary.json` | `write_evaluation_artifacts()` | 跨 trial 聚合结果、推理统计和完成标记 |

## `config.json`

对应 `RunConfig` 数据类，见 `src/coopes/cli.py:60`。它是 TOML 配置经严格校验后生成的 JSON 快照，包含 `data`、`inference` 和 `selection` 三部分。

示例配置的主要字段：

| 字段 | 示例值 | 含义 |
| --- | --- | --- |
| `data.dataset` | `arc_challenge` | 数据集名称 |
| `data.revision` | `210d026f...` | 数据集 revision |
| `data.candidate_count` | `50` | few-shot 候选池大小 |
| `data.calibration_count` | `50` | 用于估计组合效用的标定题数量 |
| `data.test_count` | `500` | 最终测试题数量 |
| `data.split_seed` | `42` | 数据切分随机种子 |
| `inference.backend` | `transformers` | 推理后端 |
| `inference.model_id` | `/home/model/Qwen2.5-7B-Instruct` | 模型路径或 ID |
| `inference.revision` | `local-d3b8...` | 模型版本 |
| `inference.device` | `cuda:0` | 推理设备 |
| `inference.dtype` | `bfloat16` | 模型精度 |
| `inference.batch_size` | `32` | 推理批大小 |
| `inference.decode.max_new_tokens` | `1` | 最大生成 token 数 |
| `inference.decode.context_window` | `8192` | 上下文窗口 |
| `selection.shot_count` | `4` | 每个 prompt 使用的示例数 `k` |
| `selection.num_samples` | `200` | 边际贡献估计的采样轮数 `M` |
| `selection.sample_seed` | `42` | 前缀采样种子 |
| `selection.random_seed` | `42` | Random 基线种子 |
| `selection.random_trials` | `10` | Random 基线 trial 数 |
| `selection.epsilon` | `0.05` | 加权 CoopES 的平滑系数 |

配置解析和校验位于 `src/coopes/cli.py:271` 的 `load_config()`。该文件用于复现实验和断点续跑，不直接参与模型推理。

## `splits.json`

由 `_splits_payload()` 生成，见 `src/coopes/cli.py:164`。它保存本次运行实际使用的三组数据：

```json
{
  "candidates": ["..."],
  "calibration": ["..."],
  "test": ["..."]
}
```

每个元素对应 `LabeledQuestion`，见 `src/coopes/data.py:48`：

```json
{
  "question": {
    "question_id": "arc_challenge:...",
    "stem": "...",
    "choices": [
      {"label": "A", "text": "..."},
      {"label": "B", "text": "..."}
    ],
    "fact": null
  },
  "answer_label": "B"
}
```

三组数据的含义：

| 集合 | 来源 | 用途 |
| --- | --- | --- |
| `candidates` | 官方 `train` split | 作为 ICL 示例候选池，并生成候选分数 |
| `calibration` | 官方 `validation` split | 计算组合效用 `V(S)` 和候选边际贡献 |
| `test` | 官方 `test` split | 在选集冻结后评估最终性能 |


## `cache.sqlite3`

持久化生成缓存，由 `open_engine()` 创建，见 `src/coopes/inference.py:601`。当前 schema 为：

```sql
CREATE TABLE generation_cache (
    cache_key TEXT PRIMARY KEY,
    response_text TEXT NOT NULL
);
```

| 字段 | 含义 |
| --- | --- |
| `cache_key` | 完整推理请求的 SHA-256，十六进制长度为 64 |
| `response_text` | 模型原始生成文本 |

缓存键不是明文 prompt，而是由以下内容共同决定：

- 推理后端、模型 ID、模型版本、dtype；
- 解码参数，如 `max_new_tokens`、`context_window`、`seed`；
- prompt template digest；
- 题目 ID；
- 有序示例 ID；
- 完整聊天消息和模板版本。

## `inference_metadata.json`

由 `open_engine()` 生成，见 `src/coopes/inference.py:601`。该文件记录实际推理环境。

主要字段：

| 字段 | 含义 |
| --- | --- |
| `backend` | `transformers` 或 `ollama` |
| `model_id` | 模型路径或模型名 |
| `model_version` | 模型 revision 或 Ollama digest |
| `runtime_versions` | Python、NumPy、PyTorch、Transformers 等版本 |
| `settings` | 设备、精度、批大小和解码设置 |
| `template_digest` | prompt 模板集合的 SHA-256 |

示例运行的关键值：

```json
{
  "backend": "transformers",
  "model_id": "/home/model/Qwen2.5-7B-Instruct",
  "model_version": "local-d3b8f5a961945310ad260269142108435f9771b72233343424f927bb39408da4",
  "template_digest": "05f1a1a4160030577241e33b6b9a94c8f97ac6856967fe13f59b63a25afaf76f"
}
```

`template_digest` 的构造见 `src/coopes/inference.py:127`。模板内容或聊天模板发生变化时，该 digest 会变化，从而影响缓存命名空间。

## `candidate_scores.csv`

由 `write_selection_artifacts()` 输出，见 `src/coopes/cli.py:414`。该文件只包含需要逐个评估候选的方法：

- `singleton`
- `coopes_uniform`
- `coopes_weighted`

`zero_shot` 没有候选示例，`random` 直接随机抽样，因此它们不会出现在该文件中。

字段如下：

| 字段 | 含义 |
| --- | --- |
| `method` | 评分方法 |
| `candidate_id` | 候选题 ID |
| `score` | 候选的效用或平均边际贡献 |
| `sample_count` | 候选被采样或评估的次数 |

### Singleton 分数

`singleton` 把每个候选单独作为 1-shot 示例，在标定集上计算准确率：

```text
score_i = V({i})
```

组合效用 `V(S)` 是由 `make_utility()` 创建的标定集准确率，见 `src/coopes/evaluation.py:113`。其计算方式是：

1. 使用 `S` 中的示例构造 prompt；
2. 对全部标定题进行预测；
3. 用解析出的答案与标定集标准答案比较；
4. 返回 `正确题数 / 标定题总数`。

Singleton 每个候选只评估一次，因此 `sample_count = 1`。示例运行中 singleton 分数范围为 `0.88` 到 `0.94`。

### CoopES 边际贡献

`coopes_uniform` 和 `coopes_weighted` 使用 `estimate_marginal_contributions()`，见 `src/coopes/selection.py:135`。

算法如下：

```text
baseline = V(空集)
重复 num_samples 次：
    从候选池无放回抽样 shot_count 个候选，保留顺序
    current = 空集
    previous = baseline
    依次加入候选 e：
        next = V(current ∪ {e})
        gain[e] += next - previous
        count[e] += 1
        previous = next

score[e] = gain[e] / count[e]
```

`score` 是该候选被加入当前上下文时带来的**平均边际贡献**。分数可以为正、为零或为负。

`sample_count` 是候选在 200 个抽样前缀中实际出现的次数。每个前缀有 4 个候选，共 800 次候选抽取，平均每个候选约 16 次。示例运行中的实际范围约为 8 到 28。

`coopes_uniform` 每步从剩余候选中等概率抽取。`coopes_weighted` 使用：

```text
weight_i = epsilon + V({i})
```

即 `singleton` 效用加 `epsilon`，见 `src/coopes/selection.py:263`。示例运行中 `epsilon = 0.05`。

最终排序规则见 `rank_candidates()`，位于 `src/coopes/selection.py:194`：

```text
按 score 降序；
score 相同时按 candidate_id 升序；
取前 shot_count 个。
```

示例运行中：

```text
coopes_uniform top-4:
  VASoL_2007_5_39
  AKDE&ED_2012_4_33
  FCAT_2012_8_7
  Mercury_7099365

coopes_weighted top-4:
  FCAT_2012_8_7
  VASoL_2007_5_39
  Mercury_400639
  AIMS_2009_4_8
```

如果某个候选的 `sample_count` 为 0，其 `score` 为空，并在 `selections.json` 的 `missing_by_method` 中报告。示例运行没有缺失候选。

## `selections.json`

由 `build_selection_plan()` 生成，见 `src/coopes/selection.py:224`，并由 `write_selection_artifacts()` 写出。

文件结构：

```json
{
  "trials": [
    {
      "method": "zero_shot",
      "trial_index": 1,
      "candidate_ids": []
    }
  ],
  "missing_by_method": {}
}
```

每个 trial 的结构对应 `SelectionTrial`，见 `src/coopes/selection.py:48`：

| 字段 | 含义 |
| --- | --- |
| `method` | `zero_shot`、`random`、`singleton`、`coopes_uniform` 或 `coopes_weighted` |
| `trial_index` | 同一方法下的 trial 编号 |
| `candidate_ids` | 冻结的 few-shot 示例 ID |


各方法含义：

- `zero_shot`：`candidate_ids` 为空，prompt 中不提供示例；
- `random`：从候选池随机抽取 `shot_count` 个候选，重复 `random_trials` 次；
- `singleton`：按单示例标定集准确率排序后取前 `k` 个；
- `coopes_uniform`：按均匀采样的平均边际贡献排序后取前 `k` 个；
- `coopes_weighted`：按加权采样的平均边际贡献排序后取前 `k` 个。


## `predictions.jsonl`

这是逐题预测文件，每行一个 JSON 对象，由 `write_evaluation_artifacts()` 写出，见 `src/coopes/cli.py:447`。

一行对应：

```text
一个 method + 一个 trial + 一道 test question
```


字段如下：

| 字段 | 含义 |
| --- | --- |
| `method` | 方法名 |
| `trial_index` | trial 编号 |
| `question_id` | 测试题 ID |
| `raw_text` | 模型原始输出 |
| `predicted_label` | 解析后的选项标签；解析失败为 `null` |
| `cache_hit` | 是否命中 SQLite 缓存 |
| `gold_label` | 标准答案标签 |
| `is_correct` | 预测标签是否等于标准答案 |

示例：

```json
{
  "cache_hit": false,
  "gold_label": "D",
  "is_correct": true,
  "method": "zero_shot",
  "predicted_label": "D",
  "question_id": "arc_challenge:ACTAAP_2007_7_3",
  "raw_text": "D",
  "trial_index": 1
}
```


## `metrics.csv`

由 `write_evaluation_artifacts()` 输出，每个 trial 一行，见 `src/coopes/cli.py:447`。

字段如下：

| 字段 | 含义 |
| --- | --- |
| `method` | 方法名 |
| `trial_index` | trial 编号 |
| `question_count` | 测试题数量 |
| `accuracy` | 答案完全匹配率 |
| `weighted_f1` | weighted F1 |
| `parse_failure_rate` | 答案解析失败率 |
| `elapsed_seconds` | 该 trial 的评估耗时 |

指标计算位于 `src/coopes/evaluation.py:157` 的 `compute_metrics()`：

```text
accuracy = 正确题数 / 总题数
parse_failure_rate = predicted_label 为 None 的题数 / 总题数
```

`weighted_f1` 使用：

```python
sklearn.metrics.f1_score(
    gold,
    predicted,
    average="weighted",
    zero_division=0,
)
```

解析失败时，预测类别使用空字符串 `""`，定义见 `src/coopes/evaluation.py:12`。

示例运行结果：

| 方法 | trial 数 | Accuracy | Weighted-F1 | Parse failure |
| --- | ---: | ---: | ---: | ---: |
| `zero_shot` | 1 | 0.894 | 0.894604 | 0.000 |
| `random` | 10 | 0.9124 | 0.915128 | 0.0044 |
| `singleton` | 1 | 0.910 | 0.912580 | 0.004 |
| `coopes_uniform` | 1 | 0.914 | 0.917527 | 0.006 |
| `coopes_weighted` | 1 | 0.908 | 0.910592 | 0.004 |

## `summary.json`

这是整个运行的最终汇总文件，由 `write_evaluation_artifacts()` 写出，见 `src/coopes/cli.py:526`。主要包含四部分。

### `method_metrics`

按方法聚合 `metrics.csv`。对于有多个 trial 的方法，对 Accuracy、Weighted-F1 和解析失败率分别取算术平均值：

```json
{
  "coopes_uniform": {
    "mean_accuracy": 0.914,
    "mean_weighted_f1": 0.9175268550096136,
    "mean_parse_failure_rate": 0.006,
    "trial_count": 1
  }
}
```


### `inference_metadata`

与 `inference_metadata.json` 内容相同，便于只打开 summary 就获得运行环境信息。

### `inference_stats`

示例运行：

```json
{
  "backend_calls": 2606,
  "cache_hits": 0,
  "retry_attempts": 0,
  "submitted_inputs": 66550
}
```

| 字段 | 含义 |
| --- | --- |
| `backend_calls` | 实际模型 batch generation 调用次数 |
| `cache_hits` | 本次运行直接命中已有缓存的请求数 |
| `retry_attempts` | 可重试推理错误的次数 |
| `submitted_inputs` | 实际提交给后端的输入数 |

统计逻辑见 `src/coopes/inference.py:392` 和 `src/coopes/inference.py:484`。

### `elapsed_seconds`

从 `run_experiment()` 开始计时到测试评估和统计完成为止。示例运行约：

```text
2782.75 秒 ≈ 46 分 23 秒
```

## 文件说明

1. `config.json` 和 `splits.json` 固定本次实验的配置和数据。
2. `cache.sqlite3` 保存选择阶段和评估阶段的所有唯一模型生成结果。
3. `candidate_scores.csv` 使用 `calibration` 和模型生成结果估计候选价值。
4. `selections.json` 根据候选分数或随机种子冻结每个 trial 的 few-shot 示例。
5. `predictions.jsonl` 在 `test` 上评估这些冻结选择。
6. `metrics.csv` 汇总单个 trial 的指标。
7. `summary.json` 对 trial 指标做方法级聚合，并记录推理成本和环境信息。

