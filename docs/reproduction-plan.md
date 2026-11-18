# CoopES 复现方案

依据论文 *Can Examples Synergize? Cooperative Example Selection for In-Context Learning* 第 III–IV 节和附录 A。本文记录从论文描述出发的实验设计，仓库结构和运行入口均为拟建内容。

## 复现目的

- 在 OpenBookQA 和 ARC-Challenge 上复现 CoopES 的核心流程。用冻结的语言模型评价示例组合，估计每个示例在不同上下文中的边际贡献，再选出分数最高的 $k$ 个示例用于推理。
- 在同一候选池、模型、提示模板和测试题上，对比随机选择、单示例评分、均匀采样 CoopES 和加权采样 CoopES。单示例评分用于检验多示例贡献是否带来额外收益，均匀与加权采样的对照用于衡量采样权重的影响。
- 保存数据划分、模型标识、采样种子、候选分数、逐题预测和运行成本，使每组结果可追溯。准确率差异以百分点报告，并记录采样种子变化时的选集重合度与性能波动。

## 实现思路

### 1. 准备数据与提示模板

| 用途 | OpenBookQA | ARC-Challenge |
| --- | --- | --- |
| 数据配置 | `allenai/openbookqa` 的 `additional` | `allenai/ai2_arc` 的 `ARC-Challenge` |
| 候选池 | 从官方 `train` 固定抽取 50 题 | 从官方 `train` 固定抽取 50 题 |
| 校准集 | 从官方 `validation` 固定抽取 50 题 | 从官方 `validation` 固定抽取 50 题 |
| 测试集 | 官方 `test` 全部 500 题 | 从官方 `test` 固定抽取 500 题 |
| 提示内容 | `fact1`、题干及选项 | 题干及选项 |

固定数据集版本和划分种子（首轮用 42），保存三个集合的题目 ID。示例展示题干、选项与正确标签，待答题展示题干和选项。OpenBookQA 的事实提示同时用于示例和待答题，所有基线使用相同输入。保留数据集原有选项标签，并以当前题目的合法标签集合解析输出。[OpenBookQA 数据卡](https://huggingface.co/datasets/allenai/openbookqa) · [ARC 数据卡](https://huggingface.co/datasets/allenai/ai2_arc)

### 2. 定义组合效用

对候选集合 $S$，按候选 ID 的固定顺序将其中的示例写入提示，调用模型回答同一批校准题。用校准集准确率定义组合效用 $V(S)$，其中 $V(\varnothing)$ 是零样本准确率。样本 $e$ 加入后的贡献为 $\Delta(e\mid S)=V(S\cup\{e\})-V(S)$。同一个示例在不同组合中的贡献可以为正、为零或为负。

推理后端由配置选择。Ollama 使用 `temperature=0`，Transformers 使用 `do_sample=False`、`num_beams=1`。两种后端分别固定模型及其版本、聊天模板、生成长度上限和推理种子。无法解析的模型回答计错并记录，服务超时等运行错误单独记录并重试。缓存按推理后端、模型版本及量化配置、提示版本、解码配置、题目 ID 和有序示例 ID 查找，支持断点续跑。

### 3. 用蒙特卡洛估计贡献并选取示例

```text
baseline = utility(空集)
重复 M 轮
    从候选池无放回抽取 k 个示例，保留抽取顺序
    当前集合 = 空集，当前效用 = baseline
    依次把每个抽到的示例加入当前集合
        新效用 = utility(加入后的集合)
        累加该示例的贡献 新效用 - 当前效用
        该示例的出现次数加一
        更新当前集合和当前效用
每个示例的分数 = 贡献总和 / 实际出现次数
按分数降序选择前 k 个示例，同分时按候选 ID 决定
```

- **均匀采样**。每一步从未入选的候选中等概率抽取，以此验证边际贡献计算和排名逻辑。
- **加权采样**。令 $w_e=\epsilon+V(\{e\})$，首轮取 $\epsilon=0.05$，每一步按剩余候选的归一化权重抽取。

两种采样都记录候选的实际出现次数。若有候选始终未出现，报告覆盖不足。需要扩大采样轮数时，单独记录新的 $M$，尤其是在做采样次数消融时。

均匀完整排列下，平均边际贡献对应标准 Shapley 值。只计算前 $k$ 个位置时，分数描述短上下文中的平均贡献。加权进一步改变了被采样的上下文分布。论文给出了加权抽样的方向，权重公式、截断估计、效用定义和提示排序均为本方案的复现约定。按分数取前 $k$ 个示例是一种选集启发式，最终组合的效果由测试集验证。

**用 SHAP 实现标准 Shapley 值**。把每个候选示例视为一个二值特征，将组合效用封装成接收一批二值向量的函数。对每个向量，取值为 `1` 的位置组成 $S$，返回对应的 $V(S)$。以单行全零向量作为 `shap.maskers.Independent` 的背景，以全一向量作为待解释输入。小候选池用 `shap.ExactExplainer` 精确计算，并逐项对照手工枚举。需要近似计算标准 Shapley 值时可用 `shap.PermutationExplainer`，并单独记录其 `max_evals`。该解释器遍历完整排列的正序和逆序，CoopES 主实验的前 $k$ 位截断和加权采样由项目自己的采样逻辑实现。[SHAP ExactExplainer](https://shap.readthedocs.io/en/latest/generated/shap.ExactExplainer.html) · [SHAP PermutationExplainer](https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html)

### 4. 分阶段运行与验收

| 阶段 | 配置 | 验收 |
| --- | --- | --- |
| 计算核对 | 6 个虚拟候选，人工定义含互补或冗余项的效用 | 枚举排列，核对截断前缀的期望贡献。完整均匀排列的每个候选分数与 `shap.ExactExplainer` 逐项对照，并核对贡献总和等于 $V(E)-V(\varnothing)$。固定采样种子时，分别核对贡献总和、出现次数与排名。 |
| 小规模试跑 | ARC-C × Qwen2.5-7B，8 个候选、8 道校准题、20 道测试题，$M=32$、$k=2$ | 生成模型预测、候选排名和测试报告。检查高低分示例是否可追溯到原始边际贡献，并用缓存重跑相同配置核对结果。 |
| 主实验 | 两个数据集 × 两个模型，50 个候选、50 道校准题、500 道测试题，$M=200$、$k=4$ | 冻结各方法的示例选择后，在同一测试集评估 Zero-shot、Random 4-shot（10 组取均值）、单示例效用 Top-4、均匀 CoopES 和加权 CoopES。报告 Accuracy、Weighted-F1、选集、解析失败率、模型请求数和用时。 |
| 消融与扩展 | ARC-C 上分别比较 $M\in\{100,150,200,250\}$ 和 $k\in\{1,2,3,4\}$ | 每次只改变一个参数，记录性能、耗时和选集重合度。扩展到论文完整基线表时，再加入 Coverage、SelfAdaptive、DataCompression 和 SE2。 |

## 技术栈

| 组件 | 选型 | 用途 |
| --- | --- | --- |
| 实验入口 | Python 3.11、`argparse`、TOML | 配置数据版本、模型、种子、$M$、$k$ 和输出目录 |
| 数据准备 | Hugging Face `datasets` | 加载官方配置并保存抽样清单 |
| 模型推理 | Ollama + `httpx`，或 Hugging Face Transformers + PyTorch | Ollama 通过 `/api/chat` 调用 `qwen2.5:7b` 和 `llama3:8b`。Transformers 用 `AutoTokenizer` 应用聊天模板，以 `AutoModelForCausalLM` 加载对应模型，再调用 `model.generate()` 批量推理。两种后端共用数据格式和答案解析。 |
| Shapley 值与采样 | SHAP、NumPy | SHAP 计算标准 Shapley 值，NumPy 实现可设种子的均匀或加权前缀采样。 |
| 评估指标 | scikit-learn | 计算 Accuracy 与 Weighted-F1。 |
| 实验记录 | SQLite、JSONL、CSV | 缓存逐题预测，导出数据划分、候选分数、选集及汇总指标 |
| 检查 | pytest | 验证边际贡献、选项解析、数据隔离和缓存命中 |

单机环境先串行运行一个模型。锁定包版本，并按后端记录运行时版本、模型版本、分词器版本及精度或量化配置。Transformers 还需记录设备及批大小。[Ollama Chat API](https://docs.ollama.com/api/chat) · [Transformers 生成 API](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)

## 代码仓库结构

```text
CoopES/
├── docs/
│   └── reproduction-plan.md    # 复现方案
├── configs/
│   ├── smoke.toml              # ARC-C × Qwen2.5-7B 小规模试跑
│   └── main.toml               # 主实验和消融配置
├── src/
│   └── coopes/
│       ├── cli.py              # 加载配置，编排数据、打分与评估
│       ├── data.py             # 数据加载、标签标准化与抽样清单
│       ├── prompts.py          # 固定模板、示例顺序与答案解析
│       ├── inference.py        # Ollama 与 Transformers 后端、运行错误与预测缓存
│       ├── selection.py        # 组合效用、前缀采样、SHAP 验证及基线选集
│       └── evaluation.py       # 测试集推理、指标与报告
├── tests/                      # 枚举核对、提示、解析与流程检查
├── runs/                       # 运行生成的清单、缓存、预测和报告
└── pyproject.toml              # 包配置、依赖与命令入口
```

执行顺序为 `data → prompts / inference → selection → evaluation`，由 `cli` 编排。每次运行在 `runs/<run_id>/` 保存配置快照、分割清单、候选分数和逐题结果，以便从汇总指标查回原始预测。

## 注意事项

- **数据隔离**。候选来自训练集，组合效用和超参数选择只使用校准集。选集冻结后才运行测试集评估。ARC-C 官方验证集只有 299 条，论文所述 500 条验证样本的具体来源无法确定。本方案从官方测试集固定抽取 500 条作为最终评估集。[ARC 数据卡](https://huggingface.co/datasets/allenai/ai2_arc)
- **提示与后端一致性**。比较不同选例方法时固定推理后端、示例顺序、选项标签、`fact1` 和答案格式。跨后端对照时单独报告结果，并记录聊天模板、模型精度与量化配置。
- **推理预算**。CoopES 前缀采样每组模型和数据集最多评估 $1+Mk$ 个前缀（去重前，含空集）。$M=200$、$k=4$、50 道校准题时，均匀版打分最多生成 40,050 个逐题输入。加权版还需最多 50 个单示例效用，可复用单示例基线缓存。SHAP 精确计算的组合数随候选规模呈指数增长，限于小候选池。用小规模试跑的实测吞吐量估算完整运行时间。
- **论文结果对照**。表 I 给出的 Accuracy 为 OpenBookQA 上 Llama3-8B 75.20%、Qwen2.5-7B 87.60%，ARC-C 上分别为 82.00% 和 88.40%。正文数值与表 I 有出入，摘要的 2.605% 也缺少明确的比较口径。以表 I 作参考，优先验证在固定实验条件下相对 Random 和单示例评分的实测变化。
