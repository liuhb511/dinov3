# D 类困难负样本优化任务清单

## 目标与范围

保留 ABCTIN_D 中独立 D 模型对小 D、弱 D 的检出能力，通过审核后的困难负样本减少误检。原 ABC/TIN 挖掘脚本和默认训练、推理流程保持可用。

本目录是本任务的唯一输出根目录：新增脚本、配置、候选、审核记录、训练派生数据、日志、权重和评估结果均存放于此。原始训练图片、GT 和基线权重通过路径引用，不移动、不覆盖；确需复制的任务数据也放在本目录。

当前状态：代码流程已实现；通过合成样本的 mask 构建、损失梯度、同步增强、滑窗和训练更新验证。尚未执行真实数据挖掘、人工审核或真实 D 权重微调，也未提交代码。

## 已实现文件

| 文件 | 功能 |
|---|---|
| `config/example.json` | 路径、GT 映射、推理、采样、损失和运行版本配置 |
| `lib/common.py` | 输出目录限制、中文路径图片读写、CSV、GT 映射、精确回贴 |
| `lib/inference.py` | 原项目 D 模型加载、权重来源选择、重叠滑窗覆盖/平均、D 概率 |
| `lib/dataset.py` | GT/负区域/ignore 同步 letterbox、翻转与旋转 |
| `lib/loss.py` | ignore-aware CE/Dice/边界损失和按图归一化困难负区域 CE |
| `scripts/mine_d_hard_negatives.py` | 微小 D 候选、分组配额、精确 mask、局部/放大/上下文图 |
| `scripts/split_d_images.py` | 原图或试样分组的训练/审计划分 |
| `scripts/prepare_d_audit.py` | 加入审计图中的 GT D 目标供人工复核，防止只审负样本 |
| `scripts/review_d_candidates.py` | 本机浏览器审核、保存和继续审核，真实 D 可标记 weak |
| `scripts/build_d_supervision_masks.py` | 精确负区域、ignore、派生 GT、冲突报告和审核快照 |
| `scripts/train_d_hard_negatives.py` | 独立 D 微调、可选加权采样、每轮权重和分项日志 |
| `scripts/evaluate_d_audit.py` | 基线/新权重局部审计、小 D/弱 D、预测和左右对比图 |
| `tests/test_workflow.py` | 合成样本与梯度验证 |
| `requirements.txt`、`.gitignore`、`lib/__init__.py` | 依赖说明、生成数据忽略规则和模块定义 |

原项目文件没有修改。调用模型时会导入原项目模型代码，任务数据不会写回原数据目录。

## 使用步骤

以下命令均在项目根目录运行，`python` 应指向已具备项目依赖的环境。先复制 `config/example.json` 为本任务的新配置，核对路径；示例数据路径只是占位，未经本机验证。各 run/version ID 必须使用未占用的名称，已有结果不允许覆盖。

```powershell
python d_hard_negative/scripts/mine_d_hard_negatives.py --config d_hard_negative/config/example.json --limit 10
```

首次用 `--limit` 检查候选，正式完整挖掘时使用新 run_id 并去掉限制。默认 `overlap_mode=overwrite` 对齐原推理；`average` 是可选独立实验。D 采用 `P(D)>d_threshold`，0.5 对齐二分类 argmax 的常规判定；概率正好相等时保留背景。

编辑挖掘输出 `images.csv` 的 `group_id`：同一试样、原图裁块及其增强副本必须同组。然后划分：

```powershell
python d_hard_negative/scripts/split_d_images.py --images d_hard_negative/data/mining/d_mining_001/images.csv
```

只有输入确为独立原图时，才使用 `--group-by-image` 替代手工分组。默认留出 20%，小数据至少需要两个独立组。请检查留出组覆盖真实小 D、弱 D 和困难负点，必要时在审核和构建前调整 split.csv；不得根据训练后的效果反复挑选留出集。

```powershell
python d_hard_negative/scripts/prepare_d_audit.py --config d_hard_negative/config/example.json --candidates d_hard_negative/data/mining/d_mining_001/candidates.csv --output data/mining/d_audit_review_001
python d_hard_negative/scripts/review_d_candidates.py --candidates d_hard_negative/data/mining/d_audit_review_001/candidates.csv --output data/reviews/d_review_001/review.csv
```

在浏览器手动打开 `http://127.0.0.1:8765`，审核窗口由本机服务提供，Ctrl+C 结束。已有输出 CSV 会继续审核；修改记录保存到同目录 `review_history.jsonl`，CSV 采用临时文件替换保存。审核编码：空=未审核，0=确认非 D，1=确认真实 D，3=不确定；TiN/ABC 对 D 二分类属于非 D，记录具体类型。审计中的 GT D 也需复核，不能自动当真值。真实弱 D 的难度填写 `weak`。

审核某个漏标目标为真实 D，只证明其存在，不代表预测轮廓就是正确标注。若要补训练 GT，填写 `positive_mask_path`：原图大小单通道 mask，非零像素为人工标注的真实 D，0 为未修正区域。未提供精确标注的漏标目标会在训练派生 GT 中忽略，不把预测轮廓自动当正标签。路径可引用原数据或本目录内的标注文件。

```powershell
python d_hard_negative/scripts/build_d_supervision_masks.py --config d_hard_negative/config/example.json
python d_hard_negative/scripts/train_d_hard_negatives.py --config d_hard_negative/config/example.json --max-steps 1
```

构建不会改原 GT。与已标 D、ignore 或正修正冲突的整个负候选被排除并忽略，详见 `conflicts.csv`；需人工解决后用新 supervision_id 重新构建。普通图片仍进入 manifest，负 mask 为零。GT=255 的区域在 CE、Dice、边界和辅助项均排除；边界监督还排除邻接 ignore 的像素。

`--max-steps 1` 是每轮最多一步的真实流程检查，必须使用独立 training_run。正式训练去掉此参数并更换 training_run。默认冻结 backbone、5 轮低学习率微调，采样权重为 1；这些是实验初值，不是已验证最佳参数。未加入自动 best 权重选择，保留每轮 checkpoint，由独立审计比较选择。所有审计图严格排除于训练。

```powershell
python d_hard_negative/scripts/evaluate_d_audit.py --config d_hard_negative/config/example.json --checkpoint d_hard_negative/runs/d_finetune_001/checkpoints/epoch_005.pth
```

配置中 checkpoint 为基线，命令参数为新模型。审计同阈值同滑窗流程，默认命中规则为预测覆盖审核目标至少 10% 像素。输出确认负目标命中数、确认真实 D 保留/漏检数、小目标（<=32像素）和 weak 分组；未审核 GT 正目标另列 `positive_gt_unverified`，不计入确认小 D/弱 D 统计。该规则是局部重叠检查，非一对一全图目标匹配，不能据此宣称完整全图召回率或精确率。预测轮廓形成的审核目标也不是精确边界真值。

## 验证结果与当前限制

```powershell
python -m unittest discover -s d_hard_negative/tests -v
python -m compileall -q d_hard_negative
```

已在 `D:/anaconda3/envs/lhb/python.exe` 环境（torch 2.5.1）通过 7 项测试：输出越界与 GT 映射、局部负 CE 梯度与全 ignore、边界 ignore 与一次训练更新、同步增强与实际 mask 构建子进程、重叠滑窗两种模式、1 像素 D 候选挖掘、训练与审计入口串联及 checkpoint 保存。语法编译也已通过。入口串联测试使用小型合成模型，未加载真实 DINOv3。

本机该环境缺少 transformers，项目内没有对应 D checkpoint；因此尚未验证真实 D 模型加载和真实数据训练，合成测试不能替代此项。依赖见 requirements.txt，需要使用可加载本项目 DINOv3 的环境。尚未生成正式任务数据或权重。人工审核和训练效果评估仍需真实数据完成。

以下勾选表示代码能力已实现或合成验证已完成；未勾选项需要真实数据、人工作业或后续实验。

## 目录规划

```text
d_hard_negative/
  README.md
  config/                 # 挖掘、mask 构建、训练和评估配置
  scripts/                # 各流程入口
  lib/                    # 本任务专用数据集、损失、采样与工具
  data/
    mining/<run_id>/       # 候选 CSV、原图局部、上下文、叠加图、component mask
    reviews/<run_id>/      # 审核 CSV、审核说明和修订记录
    masks/<version>/      # 完整尺寸负区域 mask、ignore mask、审核后的标签修正
    manifests/            # 原图路径、数据分组、样本用途和版本记录
    audit/                # 按原图留出的正负审核目标及其依据
  runs/<run_id>/
    config/               # 本次运行的固定配置及数据版本
    checkpoints/
    logs/
    evaluation/           # 预测、对比图、指标和错误记录
```

目录随实际运行创建，避免预先复制原始大数据。所有运行记录模型路径、类别映射、推理参数、输入数据清单和审核版本，禁止静默覆盖历史结果。

## 任务清单

### 1. 配置与基线

- [x] 新建本任务配置，明确独立 D checkpoint、权重来源、背景=0/D=1、原图和 GT 路径。
- [x] 将输出路径统一解析到本目录，校验禁止输出到目录外。
- [x] 固定基线预测流程，与 `infer_ABCTIN_D.py` 的模型、归一化、像素尺度、滑窗及 D 判定一致。
- [x] 记录现有滑窗覆盖行为；若修复为概率平均，作为独立实验，不与负样本训练混为一个改动。
- [x] 明确 D 类概率阈值；不再把最大概率阈值 0.1 当作有效过滤参数。

### 2. D 专用候选挖掘

- [x] 新建 `scripts/mine_d_hard_negatives.py`，借鉴根目录条状挖掘脚本，但使用真实独立 D 模型。
- [x] 在完整 D 预测上提取连通域，使用像素计数面积，不用轮廓面积替代。
- [x] 最小面积先设为可配置的 1～2 像素；取消最大边至少 16 像素及细长方向优先条件。
- [x] 保留面积、尺寸、D 平均/最大概率、原图坐标、GT 重叠和邻域信息。
- [x] 按面积及置信度分组抽取，设置每图配额并去除重复候选，避免只选大区域。
- [x] 保护已标 D 及可配置邻域；与 ABC/TIN 重叠的预测 D 进入类别混淆审核队列。
- [x] 二分类 GT 与统一多类别 GT 使用显式不同映射，禁止把统一 GT 的类别编号直接解释为 D 二分类编号。
- [x] 输出无叠加原尺寸局部、最近邻放大图、较大上下文、叠加图及精确 component mask。
- [x] 完整保留从模型预测到候选位置的对应关系，保存 component mask 后直接复用，避免重推理导致候选漂移。

### 3. 人工审核

- [x] 新建或适配本目录审核入口，展示局部和上下文，支持保存与继续审核。
- [x] 定义审核结果：确认非 D、漏标真实 D、不确定、未审核。
- [x] 对确认非 D 记录来源类型，例如普通背景、划痕、TiN、其他夹杂物；不是 D 不等于不是夹杂物。
- [x] 仅确认非 D 可进入负监督；漏标 D 经实际标注后修正正监督；不确定和未审核不得自动当负样本。
- [x] 保留审核备注、修订记录和版本；GT 未重叠只表示候选，不代表确认负样本。

### 4. 监督 mask 构建

- [x] 新建 `scripts/build_d_supervision_masks.py`，借鉴现有精确 component 回贴工具。
- [x] 生成完整图尺寸的 `d_hard_negative_mask`：1=已确认非 D，0=没有额外负监督。
- [x] 默认不膨胀、不填满 bbox；保持原始 GT 不变。
- [x] 发现已标 D 与负区域冲突时输出报告并阻止该冲突区域进入训练，待审核解决。
- [x] 为已审核的不确定目标生成显式 ignore 区域；仅明确标记的区域忽略，不将所有未标背景都忽略。
- [x] 可输出版本化派生 GT，用于已完成真实 D 标注的修正，保留来源和原始 GT。
- [x] 输出构建清单，包括审核候选数、使用数、负像素数、冲突数和数据版本。

### 5. 数据集与采样

- [x] 新建 `lib/dataset.py`，返回图像、类别 GT、困难负 mask、ignore mask 和原图身份。
- [x] 多个 mask 与图像同步缩放、裁剪、翻转；mask 使用最近邻，填充区不产生困难负监督。
- [x] 无额外 mask 的正常样本以全零旁路 mask 加载，支持继续使用原训练数据。
- [x] 初始实验保持普通采样；后续添加可配置的困难负样本采样权重。
- [x] 保留小 D、弱 D 和独立模型补回的已确认真实 D，不以困难负样本替代正常训练分布。
- [x] 已在训练集中的图片通过旁路 mask 和采样权重加强，无需重复复制图片。

### 6. 损失与训练入口

- [x] 新建 `lib/loss.py`：原 CE/Dice/边界损失 + λ × 确认非 D 区域背景 CE。
- [x] 辅助 CE 从 logits 稳定计算，只作用于审核通过且不冲突、不忽略的像素。
- [x] 先按每张图有效负区域求均值，再对包含有效负区域的图片求均值；没有有效区域时返回可反传的零。
- [x] ignore 区域同步从原 CE、Dice 和边界监督中排除，避免仅辅助项忽略、原损失仍惩罚。
- [x] 新建 `scripts/train_d_hard_negatives.py`，复用现有模型，从基线权重短程微调。
- [x] 日志分别记录原损失、辅助损失、λ、有效负像素/图片数量、学习率和采样设置。
- [x] 保存本次配置、数据版本、checkpoint 和日志到 `runs/<run_id>/`，不覆盖基线。

### 7. 审计与实验对比

- [ ] 按原始图片/试样分组，留出审核正负目标；同图相邻裁块、增强副本不得跨训练与审计分组。
- [ ] 留出目标覆盖小 D、弱 D、困难负点和类别混淆；审核结果不回灌本轮训练。
- [x] 新建 `scripts/evaluate_d_audit.py`，固定阈值、推理流程和目标匹配规则进行对比。
- [x] 输出留出负目标误报数、确认真实 D 保留/漏检数及小 D、弱 D 分组结果。
- [x] 同时输出基线/新模型局部对比图，记录新的错误类型；不把局部审计指标宣称为完整全图召回率。
- [ ] 按顺序实验：基线 → 仅区域辅助损失 → 仅提高采样率 → 二者组合。其他条件尽量一致。
- [x] 无可信全图 GT 时，按留出审核结果和可见召回损伤选择候选权重，不仅依据训练 loss 或原 mIoU。

### 8. 必要验证与交付

- [ ] 用少量图片跑通挖掘、审核、mask 构建、数据加载和一次训练更新。
- [x] 验证小点候选能被保留、mask 回贴坐标正确、不同 GT 映射正确。
- [x] 验证几何增强后多个 mask 仍对齐，ignore 和冲突像素确实不受负惩罚。
- [x] 验证辅助损失只改变指定区域的监督，无负区域时正常训练，正样本标签保持有效。
- [x] 完成 README 使用说明、配置示例、数据格式及运行顺序。
- [ ] 每轮训练后只在训练来源中挖掘新的残余误检，审核后形成下一版本；保留审计集独立。

## 推荐实施顺序

1. 配置与基线 → D 候选挖掘 → 人工审核。
2. 划分训练/留出审核图片 → 构建精确监督 mask。
3. 接入数据集与区域辅助损失 → 少量样本验证。
4. 保持采样不变，完成第一轮短程微调与审计。
5. 收益明确后，再测试采样强化及两者组合。

第一轮验收重点：留出的确认负点误报减少，同时已确认真实 D，尤其小 D、弱 D，没有明显退化。可接受的召回损伤及误检减少幅度在取得基线审计结果后确定，不预先虚构指标。
