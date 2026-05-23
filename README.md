# 知识编辑实验

本项目是 LLM 知识编辑实验的代码。
当前保留的内容覆盖了本次实验的主要环节，包括基线测试、ROME 事实编辑、MEMIT 批量编辑，以及对已有结果文件的汇总评估。

## 文件

必需的入口脚本：

- `baseline.py`：运行基线测试，用于获得编辑前模型在任务上的原始表现。
- `edit_rome.py`：运行 ROME 单事实编辑实验。
- `edit_memit.py`：运行 MEMIT 批量知识编辑实验。
- `evaluate.py`：对已有结果文件进行汇总和指标整理。

核心实现：

- `scripts/baseline_eval.py`：基线评测逻辑。
- `scripts/run_rome_single.py`：ROME 单样本编辑的主要执行逻辑。
- `scripts/run_memit_batch.py`：MEMIT 批量编辑的主要执行逻辑。
- `scripts/summarize_metrics.py`：结果汇总与指标统计逻辑。
- `scripts/common.py`：公共工具函数与共享代码。

配置文件：

- `configs/rome_qwen2.5_0.5b.yaml`：ROME 实验使用的超参数配置。
- `configs/memit_qwen2.5_0.5b.yaml`：MEMIT 实验使用的超参数配置。

数据集：

- `custom_10.json`：主实验中使用的小规模样例数据。
- `custom_10_crosslingual.json`：跨语言附加实验数据。
- `data/knowedit/ZsRE/ZsRE-test-all.json`：保留的 ZsRE 测试数据。
- `data/knowedit/ZsRE/zsre_500_final.json`：MEMIT 批量编辑实验所用数据。
- `data/knowedit/ZsRE/zsre_stats_corpus.jsonl`：统计信息相关语料文件。

预计算的 MEMIT 统计数据：

- `easyedit_data/stats/Qwen2.5-0.5B/...`

随包附带的依赖：

- `EasyEdit/`

实验报告：

- `SZ2516076-陈艺爽-03-KnowledgeEditing.pdf`：完整的实验报告。

## 环境

在项目根目录安装依赖：

```powershell
pip install -r requirements.txt
```

说明：

1. `requirements.txt` 复用了 `EasyEdit/requirements.txt`。
2. 本地 `EasyEdit` 文件夹通过 `sys.path` 导入；不需要 editable install。
3. 本包不包含模型权重。权重必须已存在于 Hugging Face 缓存中，或在运行环境中下载。
4. 如果使用文中的默认命令，运行环境需要能够访问 `cuda:0`；如果只使用 CPU 或其他 GPU，需要自行调整 `--device` 参数。

## 运行命令

以下所有命令均假定：

1. 当前工作目录为项目根目录。
2. 所需依赖已经安装完成。
3. 对应模型权重已经可用。
4. 输出结果会按脚本逻辑写入 `results/` 下的相应目录中。

### 任务 1：基线

该步骤用于获得模型在未编辑状态下的表现，通常可作为后续编辑实验的对照结果。

```powershell
python baseline.py --data_path custom_10.json --device cuda:0
```

### 任务 2：ROME 单事实编辑

该步骤用于执行单条知识的定点编辑，并观察编辑是否成功以及是否影响相关泛化行为。

完整的 10 样本运行：

```powershell
python edit_rome.py --data_path custom_10.json --hparams_path configs/rome_qwen2.5_0.5b.yaml --device cuda:0
```

单案例展示：

```powershell
python edit_rome.py --data_path task2_showcase_kazakhstan.json --hparams_path configs/rome_qwen2.5_0.5b.yaml --device cuda:0
```

### 任务 3：MEMIT 批量编辑

该步骤用于对较大规模样本进行批量知识编辑，适合展示多条知识同时修改时的整体效果。

```powershell
python edit_memit.py --data_path data/knowedit/ZsRE/zsre_500_final.json --hparams_path configs/memit_qwen2.5_0.5b.yaml --sample_size 500 --seed 42 --device cuda:0 --edit_batch_size 500
```

### 任务 4：评估已有结果

如果不重新运行实验，也可以直接对现有结果文件做汇总分析。

汇总单个结果文件：

```powershell
python evaluate.py --path results/memit/memit_batch_20260519_220300.json
```

其他保留的结果文件：

- `results/baseline/baseline_20260519_191521.json`
- `results/rome/rome_single_20260519_210423.json`
- `results/rome_task2_showcase/rome_single_20260519_203839.json`
- `results/baseline_crosslingual/baseline_20260519_185045.json`
- `results/rome_crosslingual/rome_single_20260519_185306.json`

## 保留的结果

精简后的提交包为每个主要实验阶段保留一个最终结果文件，以及可选的加分项任务5的输出：

- 基线：`results/baseline/baseline_20260519_191521.json`
- ROME：`results/rome/rome_single_20260519_210423.json`
- MEMIT：`results/memit/memit_batch_20260519_220300.json`
- 任务 2 展示：`results/rome_task2_showcase/rome_single_20260519_203839.json`
- 任务 5 基线：`results/baseline_crosslingual/baseline_20260519_185045.json`
- 任务 5 ROME：`results/rome_crosslingual/rome_single_20260519_185306.json`

