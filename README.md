# CoopES

## 项目介绍

CoopES 是论文 *Can Examples Synergize? Cooperative Example Selection for In-Context Learning* 的复现项目。它研究如何为大语言模型的少样本提示选择一组示例。

程序从训练集抽取候选示例，以校准集准确率衡量示例组合的效用。每次采样依次加入示例，记录准确率的变化，再按候选在不同组合中的平均边际贡献选出前 `k` 个示例。选集冻结后，在同一独立测试集上比较三种方法。

| 方法 | 示例选择 |
| --- | --- |
| Zero-shot | 提示中只有任务说明和待答题 |
| CoopES Uniform | 均匀地无放回抽取候选，按平均边际贡献排名 |
| CoopES Weighted | 以单示例的校准集准确率加平滑常数作为采样权重，按平均边际贡献排名 |

主实验使用 Transformers 推理，Ollama 也可作为推理后端。实验报告记录准确率、加权 F1 和时间开销。

## 仓库代码结构

```text
CoopES/
├── configs/                  # 小规模试跑、主实验和完整复现实验的 TOML 配置
├── docs/
│   ├── reproduction-plan.md   # 复现方案与实验约定
│   ├── run-artifacts.md       # 运行产物和字段说明
│   ├── plans/                # 模块详细设计
│   └── experiments/          # 历史实验报告与汇总 CSV
├── src/coopes/
│   ├── cli.py                # 命令行入口、实验编排与产物写入
│   ├── data.py               # 数据集加载、题目标准化和采样
│   ├── prompts.py            # 提示构建与答案解析
│   ├── inference.py          # 推理后端、重试和 SQLite 缓存
│   ├── selection.py          # 前缀采样、候选排名与 SHAP 核对
│   └── evaluation.py         # 校准集效用、测试集预测和指标
├── tests/                    # 单元测试与实验流程测试
├── pyproject.toml            # 打包配置、依赖和 CLI 注册
└── README.md
```


## 快速启动

### 1. 安装依赖并准备模型

项目需要 Python 3.11 或更高版本。在实验服务器上，从仓库根目录创建并激活虚拟环境。

| 环境 | 创建虚拟环境 | 激活虚拟环境 |
| --- | --- | --- |
| Windows PowerShell | `python -m venv .venv` | `.\.venv\Scripts\Activate.ps1` |
| macOS / Linux | `python3 -m venv .venv` | `source .venv/bin/activate` |

若 PowerShell 禁止执行激活脚本，可直接用 `.\.venv\Scripts\python.exe` 和 `.\.venv\Scripts\coopes.exe` 执行下文的 `python` 和 `coopes` 命令。

安装数据加载和 Transformers 推理依赖。

```sh
python -m pip install -e ".[data,transformers]"
```

主实验配置使用服务器上的本地权重。根据实际环境设置 `[inference]` 中的 `model_id`、`revision`、`device`、`dtype` 和 `batch_size`。数据集首次加载时从 Hugging Face 获取。

### 2. 运行小规模实验（可选）

试跑配置默认从 Hugging Face 加载 Qwen2.5-7B-Instruct。服务器已有权重时，可将 `model_id` 改为本地模型目录并更新对应的 `revision`。

```sh
coopes --config configs/smoke-transformers.toml --run-dir runs/smoke-transformers
```

该配置使用 8 个候选示例、8 道校准题和 20 道测试题，两种 CoopES 方法各选出 2 个示例。命令完成后会打印 `summary.json` 的路径和已评估的方法数。可从以下文件查看结果。

| 文件 | 内容 |
| --- | --- |
| `summary.json`、`metrics.csv` | 各方法的汇总指标、推理统计和逐轮指标 |
| `candidate_scores.csv`、`selections.json` | 候选得分、采样次数和最终选集 |
| `predictions.jsonl` | 测试集逐题输出、解析标签和判分 |
| `config.json`、`splits.json`、`inference_metadata.json` | 实验配置、数据划分快照和推理环境 |
| `cache.sqlite3` | 已生成的模型响应 |

中断后可用相同配置和运行目录重新执行命令，复用已有缓存。目录已有 `summary.json` 时，请为新一轮实验指定新的 `--run-dir`。

### 3. 运行 Transformers 主实验

每组实验使用 50 个候选示例、50 道校准题和 500 道测试题。两种 CoopES 方法各抽样 200 轮并选出 4 个示例。

| 数据集 | Qwen2.5-7B-Instruct | Meta-Llama-3-8B-Instruct |
| --- | --- | --- |
| ARC-Challenge | [`full-arc_challenge-qwen.toml`](configs/full-arc_challenge-qwen.toml) | [`full-arc_challenge-llama3.toml`](configs/full-arc_challenge-llama3.toml) |
| OpenBookQA | [`full-openbookqa-qwen.toml`](configs/full-openbookqa-qwen.toml) | [`full-openbookqa-llama3.toml`](configs/full-openbookqa-llama3.toml) |

以 ARC-Challenge 和 Qwen2.5-7B-Instruct 为例。

```sh
coopes --config configs/full-arc_challenge-qwen.toml --run-dir runs/full-arc_challenge-qwen
```

### 4. 自定义实验配置

复制一份 Transformers TOML 配置。以下示例以小规模试跑配置为模板，正式实验可从上表选择对应的数据集和模型配置。

| 环境 | 复制配置 |
| --- | --- |
| Windows PowerShell | `Copy-Item configs/smoke-transformers.toml configs/custom.toml` |
| macOS / Linux | `cp configs/smoke-transformers.toml configs/custom.toml` |

在 `configs/custom.toml` 中按实验目标修改参数。

| 配置段 | 常用参数 |
| --- | --- |
| `[data]` | `dataset` 选 `arc_challenge` 或 `openbookqa`，切换时一并更新对应的 `revision`。用 `candidate_count`、`calibration_count`、`test_count` 设置抽样规模，用 `split_seed` 控制数据抽样。 |
| `[inference]` | `model_id` 可填本地模型目录或 Hugging Face 模型 ID，`revision` 记录权重版本。`device`、`dtype` 和 `batch_size` 按服务器硬件设置。 |
| `[inference.decode]` | `max_new_tokens` 限制生成的 token 数，`context_window` 设置上下文窗口大小。 |
| `[selection]` | `shot_count` 是最终选出的示例数，最多等于 `candidate_count`。`num_samples` 控制采样轮数，`sample_seed` 控制选例采样，`epsilon` 是加权采样的平滑常数。 |

例如，将 `shot_count` 改为 `3`、`num_samples` 改为 `64`，然后运行一组 3-shot 实验。

```sh
coopes --config configs/custom.toml --run-dir runs/custom
```

如果运行报告候选覆盖不足，可提高 `num_samples`，再指定新的输出目录运行。

### 5. 使用 Ollama 后端（可选）

安装 Ollama 后，确认服务监听 `http://127.0.0.1:11434`。如果服务尚未运行，先在另一个终端执行 `ollama serve` 并保持运行。 [Ollama 下载](https://ollama.com/download)

```sh
python -m pip install -e ".[data,ollama]"
ollama pull qwen2.5:7b
coopes --config configs/smoke.toml --run-dir runs/smoke-ollama
```

Ollama 的完整规模配置保存在 [`configs/main.toml`](configs/main.toml)。可通过 `base_url` 和 `model_tag` 指定 Ollama 服务与模型。复现时可将 `model_digest` 固定为 `/api/tags` 返回的完整摘要。

### 6. 运行开发测试（可选）

执行 `python -m pip install -e ".[dev,ollama,transformers,shap]"` 安装测试依赖，再运行 `python -m pytest` 和 `python -m ruff check src tests`。测试使用模拟推理响应，无需下载模型权重。

## 参考资料

- 方法背景、效用定义、采样规则和实验约定。 [CoopES 复现方案](docs/reproduction-plan.md) · [模块详细设计](docs/plans/2026-11-18-coopes-detailed-design.md)
- 输出字段、缓存和续跑机制。 [运行产物说明](docs/run-artifacts.md)
- 历史实验包含 Qwen2.5-7B-Instruct 和旧版 Llama 3 基座模型，后续报告记录了 Llama 3 Instruct 的完整结果。 [Transformers 完整实验](docs/experiments/2026-11-18-full-transformers-results.md) · [Llama 3 Instruct 实验](docs/experiments/2026-11-19-llama3-instruct-results.md)
- 数据集说明。 [OpenBookQA](https://huggingface.co/datasets/allenai/openbookqa) · [ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc)
- 推理工具和 Shapley 值核对。 [Transformers 生成 API](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation) · [Ollama CLI](https://docs.ollama.com/cli) · [Ollama 模型列表 API](https://docs.ollama.com/api/tags) · [SHAP ExactExplainer](https://shap.readthedocs.io/en/latest/generated/shap.ExactExplainer.html)
