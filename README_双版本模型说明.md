# DINOv3 非金属夹杂物识别项目说明

本项目目前同时保留两套模型路线：

1. **原 DINO 版本（Legacy）**：早期方案，继续用于历史模型复现、基线对比和已有推理流程。
2. **`inclusion_v2` 版本（当前主线）**：共享 DINOv3 特征提取，并使用 Gate + Strip/Point 双专家，作为后续主要优化方向。

> 原则：旧版本尽量保持稳定，不直接修改原网络文件；新功能和新实验优先放在 `inclusion_v2/`。

---

## 1. 两个版本的定位

### 1.1 原 DINO 版本（Legacy）

原版本采用较早的任务拆分方式，Strip 与 Point 类别分别训练/推理，重型 DINOv3 特征提取不共享。

主要用途：

- 复现历史结果；
- 作为 `inclusion_v2` 的 baseline；
- 保留已有训练权重与推理流程；
- 新版本出现性能回退时用于对照。

原版本原则上只做必要的兼容和 bug 修复，不再作为主要结构优化对象。

### 1.2 `inclusion_v2` 版本（当前主线）

当前 MVP 结构：

```text
Input 1024×1024
        │
        ▼
DINOv3 ViT-B/16
        │
   L4 / L8 / L12
        │
        ▼
Lightweight Feature Fusion
        │
        ▼
Shared Decoder
        │
        ├── Gate Head
        │    ├─ Background
        │    ├─ Strip
        │    └─ Point
        │
        ├── Strip Head
        │    ├─ Background
        │    ├─ A
        │    ├─ B
        │    ├─ C
        │    ├─ TIN-B/TIN-C
        │    ├─ TIN-D
        │    ├─ HH
        │    ├─ XW
        │    └─ XQL
        │
        └── Point Head
             ├─ Background
             ├─ D
             ├─ HC
             └─ SZ
```

核心目标是：**一套 DINOv3 特征同时服务 Strip 和 Point，减少重复计算，同时保留不同类别族的专门判别能力。**

---

## 2. 统一类别编码

| ID | 类别 |
|---:|---|
| 0 | Background |
| 1 | A |
| 2 | B |
| 3 | C |
| 4 | D |
| 5 | TIN-B / TIN-C |
| 6 | TIN-D |
| 7 | HH |
| 8 | XW |
| 9 | XQL |
| 10 | HC |
| 11 | SZ |

说明：

- JSON 中 `TIN-B`、`TIN-C`、`TIN-B/TIN-C` 均统一映射为类别 5；
- `TIN-D` 属于 Strip 分支；
- `HH / XW / XQL` 属于 Strip 家族；
- `HC / SZ` 属于 Point 家族。

---

## 3. `inclusion_v2` 局部类别映射

### Gate

```text
0 = Background
1 = Strip
2 = Point
```

### Strip Head

```text
0 = Background
1 = A
2 = B
3 = C
4 = TIN-B/TIN-C
5 = TIN-D
6 = HH
7 = XW
8 = XQL
```

### Point Head

```text
0 = Background
1 = D
2 = HC
3 = SZ
```

---

## 4. 数据与分辨率

当前训练与推理主要使用：

```text
训练 patch：1024 × 1024
推理窗口：1024 × 1024
```

原始大图约：

```text
2448 × 2048
```

当前像素尺度约：

```text
0.5448 μm / px
```

大图推理采用滑窗，重叠区域建议累计概率并除以 count map。

---

## 5. 当前项目结构

```text
dinov3/
│
├── dinov3_model/              # DINOv3 backbone / 官方相关代码
├── models/                    # 原 DINO 版本模型
├── losses/                    # 原版本 loss
├── utils/                     # 原版本/通用工具
│
├── inclusion_v2/              # 当前主线
│   ├── data/
│   │   └── dataset.py
│   ├── models/
│   │   ├── encoder.py
│   │   ├── model.py
│   │   └── ...
│   ├── losses/
│   ├── utils/
│   └── tools/
│
├── v2_train_inclusion.py
├── v2_infer_inclusion.py
├── v2_compare_baseline.py
├── evaluate_inclusion_diagnostics.py
└── ...
```

---

## 6. 当前 `inclusion_v2` 采样规则

当前 Dataset：

```python
def __len__(self):
    return len(self.image_list)

def __getitem__(self, idx):
    image, mask, img_name = self._load(idx)
```

Dataset 内部没有额外随机重采样。

当前 DataLoader：

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN["batch_size"],
    shuffle=True,
    num_workers=TRAIN["num_workers"],
    pin_memory=True,
    drop_last=True,
)
```

因此当前规则是：

> **每个 epoch 对全部训练 patch 随机打乱，每个 patch 基本出现 1 次。**

当前没有：

- `WeightedRandomSampler`
- hard-negative 重采样
- 小目标专项采样
- 类别平衡采样

---

## 7. `inclusion_v2` 融合方式

最终类别分数：

```text
score(class)
=
P_expert(class)
×
P_gate(group)^alpha
```

当前默认：

```text
alpha = 0.5
```

这里的 score 是融合分数，不应直接理解为严格校准概率。

现有 TIN-D 诊断显示，最终 fusion 不是当前 TIN-D 漏检的主要瓶颈，因此暂不优先调整 alpha。

---

## 8. 当前代表性评估结果

| 指标 | epoch62 | epoch64 |
|---|---:|---:|
| Inclusion Precision | 0.866 | 0.831 |
| Inclusion Recall | 0.697 | 0.765 |
| Inclusion F1 | 0.772 | 0.797 |
| TIN-D pixel recall | 0.752 | 0.662 |
| TIN-D final object recall | 0.821 | 0.701 |
| D small recall | 0.708 | 0.632 |
| D medium recall | 0.768 | 0.758 |
| D large recall | 0.824 | 0.853 |

结论：

- epoch64：整体 Recall / F1 更高，但更激进；
- epoch62：Precision、TIN-D、小 D 更好；
- 两个 checkpoint 都有保留价值。

---

## 9. 当前主要问题

### 9.1 Strip 假阳性

epoch64：

```text
BG → Strip apparent FP components = 9271
总面积 ≈ 324050 px
```

B/C 的大面积误检更突出，很多误检呈细长、竖直结构。

说明模型容易把“细长 + 沿轧制方向”的背景结构误认为 A/B/C。

### 9.2 小目标漏检

TIN-D：

```text
≤5 μm      final object recall ≈ 0.423
5–7 μm                          ≈ 0.749
7–10 μm                         ≈ 0.852
>15 μm                          ≈ 1.000
```

D 也存在明显尺寸效应。

结合 ViT-B/16，可判断当前模型对约 7～10 px 级目标的细节保留不足。

### 9.3 TIN-D 漏检主要发生在 Gate 和 Strip Expert

epoch64 的 TIN-D 最终漏检中：

```text
约 59.6%：Gate 失败
约 40.0%：Gate 通过，但 Strip Expert 失败
约 0.4%：最终 fusion 才失败
```

优先级：

1. 小 TIN-D Gate 识别；
2. Strip Expert 中 TIN-D / TIN-BC 区分；
3. 最后才考虑 fusion alpha。

### 9.4 主要细分类混淆

当前主要是：

```text
B ↔ C
TIN-D ↔ TIN-B/TIN-C
```

epoch64：

```text
B → C ≈ 12.9%
C → B ≈ 12.8%
A → C ≈ 0.9%
```

---

## 10. Hard Negative 流程

目的：降低 BG 被误判为 A/B/C 的假阳性。

流程：

```text
训练集
  ↓
使用较激进 checkpoint 推理
  ↓
完整 A/B/C 预测连通域
  ↓
与 GT 前景重叠 → 排除
距离 GT 前景过近 → 排除
面积 < 64 px → 排除
最大边 < 16 px → 排除
  ↓
人工审核
```

人工审核标签：

```text
0 = 确认负样本
1 = 漏标夹杂物
2 = 已知干扰物
3 = 不确定
```

第一版训练只使用：

```text
0 = 确认负样本
```

并生成独立：

```text
hard_negative_masks/
```

原始 GT 不修改。

---

## 11. Hard Negative 的训练使用方式

### 11.1 HN patch 重采样

建议：

```text
普通 patch = 1.0
包含人工确认 HN 的 patch = 2.0
```

后续通过 `WeightedRandomSampler` 实现。

建议保持：

```python
num_samples = len(train_dataset)
replacement = True
```

这样一个 epoch 的总采样数不变，但 HN patch 被抽中的概率约为普通 patch 的 2 倍。

### 11.2 Strip Head Hard-Negative Loss

在 HN 区域定义：

```text
P_inc_strip =
P(A) + P(B) + P(C) + P(TINBC) + P(TIND)
```

损失：

```text
L_hn = -log(1 - P_inc_strip)
```

第一版建议：

```text
lambda_hn = 0.1
```

不要求 HN 一定预测为 Background，只要求它不要被预测为真正的 Strip inclusion。

因此模型仍可选择：

```text
Background / HH / XW / XQL
```

第一版不建议强制 Gate 输出 Background。

---

## 12. 后续优化路线

### MVP-2A：训练稳定性

```text
当前模型
+ Differential LR
+ EMA
```

建议：

```text
DINO backbone       ≈ 0.1 × head LR
fusion / decoder    ≈ 0.5 × head LR
heads               = 1.0 ×
```

### MVP-2B：Hard Negative

```text
MVP-2A
+ HN patch weight = 2.0
+ lambda_hn = 0.1
```

目标：

```text
Precision ↑
大面积 BG→B/C FP ↓
Recall 基本不下降
```

### MVP-2C：Small Object

建议 size-aware：

```text
TIN-D:
≤5 μm  ×2.0
5–7 μm ×1.5

D:
≤5 μm  ×1.5
5–7 μm ×1.25
```

并同时作用于相应 Gate 与 Expert loss。

### MVP-2D：Object Consistency

针对：

```text
A / B / C / TINBC / TIND
```

增加轻量 object-level consistency loss。

建议初始：

```text
lambda_obj = 0.05
```

### MVP-3：高分辨率细节分支

只有当完成上述低成本优化后，小 TIN-D / 小 D 仍明显受尺寸限制，再考虑：

```text
轻量 1/4 分辨率 detail branch
→ shared decoder
```

当前不建议优先上：

- 第二套重型 backbone；
- 更大的 DINO；
- deformable decoder；
- 重型 attention；
- 独立 verifier 网络。

---

## 13. 推荐实验顺序

```text
Experiment A
当前 baseline
+ Differential LR
+ EMA

Experiment B
A
+ Hard Negative sampling
+ Hard Negative loss

Experiment C
B
+ small-object sampling
+ small-object weighting

Experiment D
C
+ object consistency
```

每次尽量只增加一类主要变量，方便判断收益来源。

---

## 14. 后续评估重点

不要只看 Inclusion F1，至少同时监控：

```text
Inclusion Precision
Inclusion Recall
Inclusion F1

TIN-D final object recall
TIN-D ≤5 μm recall
D small recall

BG→Strip apparent FP
面积 ≥64 px 的 BG→A/B/C FP
面积 ≥256 px 的 BG→A/B/C FP

B↔C confusion
TIN-D↔TINBC confusion
```

---

## 15. 两套模型的使用原则

### 原 DINO

用于：

- 历史结果复现；
- baseline；
- 已有旧模型部署；
- 与新版本进行速度/性能对比。

原则：

> **稳定保留，尽量不改。**

### `inclusion_v2`

用于：

- 当前主要训练；
- 新 loss；
- Hard Negative；
- small-object 优化；
- Gate / Expert 诊断；
- 后续结构实验。

原则：

> **后续优化集中在 `inclusion_v2`，同时保持与旧版可比较。**

---

## 16. 当前总体判断

目前没有证据需要放弃 `inclusion_v2`。

当前主要瓶颈已经比较明确：

```text
Strip 假阳性
→ Hard Negative

小目标漏检
→ size-aware training
→ 必要时再加轻量高分辨率分支

训练后期漂移
→ Differential LR + EMA

细分类问题
→ B↔C、TIND↔TINBC
→ class balance / object consistency
```

当前更合理的路线是：

> **先优化训练信号、困难负样本、小目标和训练稳定性，再判断是否需要升级网络结构。**
