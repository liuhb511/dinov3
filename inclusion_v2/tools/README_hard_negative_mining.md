# Hard Negative Mining 工具链

这一阶段**不修改网络、不修改 loss、不修改原始 GT**。

目标是先建立一批经过人工确认的、真正可靠的 Strip hard negatives。

## 目录文件

- `mine_hard_negatives.py`
  - 在训练集上使用当前 checkpoint 推理
  - 自动寻找 `GT=BG && Pred in {A,B,C}`
  - 默认筛 `area >= 64 px` 且 `max(w,h) >= 16 px`
  - 保存 raw crop、overlay crop、精确 component mask
  - 生成 `candidates.csv`

- `review_hard_negatives.py`
  - OpenCV 按键审核
  - `0 = CONFIRMED_NEGATIVE`
  - `1 = MISSED_INCLUSION`
  - `2 = KNOWN_DISTRACTOR`
  - `3 = UNCERTAIN`
  - `S = skip`
  - `B = back`
  - `Q/ESC = 保存退出`
  - 每次按键都会立即写回 CSV

- `build_hard_negative_masks.py`
  - 默认只使用 `review_label=0`
  - 生成独立 1024×1024 二值 `hard_negative_masks/*.png`
  - `0=普通区域, 1=确认 hard negative`
  - 不修改原始 GT
  - 同时拆分导出：
    - `confirmed_hard_negatives.csv`
    - `missed_inclusions.csv`
    - `known_distractors.csv`
    - `uncertain.csv`

## 推荐使用步骤

### 1. 放置文件

建议放到仓库：

```text
inclusion_v2/tools/
├── mine_hard_negatives.py
├── review_hard_negatives.py
└── build_hard_negative_masks.py
```

或者暂时放仓库根目录也可以。

### 2. 修改 mine 配置

重点修改：

```python
"train_image_dir": ".../train/images",
"train_mask_dir":  ".../train/masks",
"checkpoint":      "...epoch64....pth",
```

模型结构字段必须与训练一致。

第一次建议：

```python
"min_component_area_px": 64,
"min_max_side_px": 16,
"max_candidates_per_class": 500,
```

程序按面积从大到小保存，所以即使每类生成 500 个，人工也可以只审核前 30~50 个。

### 3. 挖候选

在项目根目录：

```bash
python inclusion_v2/tools/mine_hard_negatives.py
```

或者按你的实际文件位置运行。

### 4. 人工审核

```bash
python inclusion_v2/tools/review_hard_negatives.py
```

第一轮建议只审核：
- A：前 30~50
- B：前 30~50
- C：前 30~50

如果看到的候选中大量是真正的漏标夹杂物，就先不要把 apparent FP 强化为 background。

### 5. 生成 hard-negative masks

```bash
python inclusion_v2/tools/build_hard_negative_masks.py
```

输出：

```text
hard_negative_mining/
├── confirmed_hard_negatives.csv
├── missed_inclusions.csv
├── known_distractors.csv
├── uncertain.csv
└── hard_negative_masks/
    ├── xxx.png
    └── ...
```

## 下一阶段如何用于训练

先不要现在接入。

等人工审核出几十到一百多个候选后，先看：

```text
CONFIRMED_NEGATIVE 比例
MISSED_INCLUSION 比例
KNOWN_DISTRACTOR 比例
UNCERTAIN 比例
```

如果 confirmed negative 数量足够且可信，再做训练侧：

1. 含 confirmed hard negative 的 patch sampling weight ≈ 2.0
2. Strip Head 增加轻量 rejection：
   `P_inc_strip = P(A)+P(B)+P(C)+P(TINBC)+P(TIND)`
3. 仅在 hard-negative mask 上惩罚高 `P_inc_strip`
4. 初始 `lambda_hardneg = 0.1`
5. 不直接修改 Gate，也不修改原 GT

这能避免把漏标的真实夹杂物错误训练成 background。
