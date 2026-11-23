# CoopES 仓库协作指南

CoopES 是用于复现上下文学习中协作式示例选择的 Python 项目。它在 OpenBookQA 和 ARC-Challenge 上比较 Zero-shot、CoopES Uniform 和 CoopES Weighted，支持 Ollama 与 Transformers 推理后端。运行入口是 `coopes`，由 `pyproject.toml` 注册到 `coopes.cli:main`。

## 代码导航

如果仓库根目录存在 `.codegraph/`，定位代码、读取源码或梳理调用链时，先使用 CodeGraph。优先使用 `codegraph_explore` MCP 工具，在查询中指定文件或符号名。MCP 工具不可用时，从仓库根目录运行下面的命令。

```sh
codegraph explore "run_experiment load_config build_selection_plan"
```

CodeGraph 未覆盖的内容可用 `rg` 补查。仓库没有 `.codegraph/` 时，直接用 `rg` 和 `rg --files`，由仓库维护者决定是否建立索引。

| 路径 | 职责 |
| --- | --- |
| `src/coopes/cli.py` | TOML 校验、实验编排、续跑校验和运行产物写入 |
| `src/coopes/data.py` | Hugging Face 数据集加载、题目标准化和确定性抽样 |
| `src/coopes/prompts.py` | 组装对话提示，按题目的合法选项解析回答 |
| `src/coopes/inference.py` | Ollama 与 Transformers 推理、重试、模型元数据和 SQLite 缓存 |
| `src/coopes/selection.py` | 均匀及加权前缀采样、候选排名和可选的 SHAP 核对 |
| `src/coopes/evaluation.py` | 校准集效用、测试集判分和指标计算 |
| `tests/` | 模块测试与实验流程测试 |
| `configs/` | 试跑、主实验和完整实验的 TOML 配置 |
| `docs/` | 复现方案、详细设计、产物说明和历史实验结果 |

## 开发与验证

使用 Python 3.11 或更高版本，在仓库根目录的虚拟环境中运行下列命令。完整测试需要安装两种推理后端和 SHAP 依赖，测试用的推理响应由代码模拟。

```sh
python -m pip install -e ".[dev,ollama,transformers,shap]"
python -m pytest
python -m ruff check src tests
```

Python 静态检查按 `pyproject.toml` 中的 Ruff 配置执行。实现改动优先运行相关的 `tests/test_*.py`，变更涉及多个模块时运行全量测试。配置、CLI 参数或产物格式有变动时，同步核对 README、示例配置和对应的文档。

需要验证真实推理时，安装 `data` 和对应后端的可选依赖，并按照 README 准备数据和模型。Ollama 小规模试跑入口如下。

```sh
coopes --config configs/smoke.toml --run-dir runs/smoke
```

## 实验约定

- 从官方训练集抽候选，从验证集抽校准题，从测试集抽评估题。用校准集准确率估计组合效用，冻结选集后再评估测试集。
- 保留数据的原始选项标签和顺序。提示中的示例按题目 ID 固定排序，候选评分同分时也按 ID 排序。SHAP 接口用于标准 Shapley 值核对，主实验使用 `selection.py` 的前缀采样。
- 复现实验时记录数据集版本、模型版本、划分和采样种子。改变参数时复制现有 TOML 配置并用新的 `--run-dir`。同一配置的中断运行可以复用原目录和 `cache.sqlite3`，已有 `summary.json` 的目录视为运行完成。
- 将本地运行产物保存在 Git 忽略的 `runs/`。仓库中的实验报告和汇总 CSV 位于 `docs/experiments/`。

## 项目文档

- 安装、启动和自定义实验配置。 [README](README.md)
- 复现约定与模块接口。 [复现方案](docs/reproduction-plan.md) · [详细设计](docs/plans/2026-11-18-coopes-detailed-design.md)
- 数据快照、缓存、评分和指标字段。 [运行产物说明](docs/run-artifacts.md)
