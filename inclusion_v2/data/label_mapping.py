"""
inclusion_v2/data/label_mapping.py

统一类别编码（12 类 + 背景）—— 全链路（mask / target / loss / metrics / 推理）共用：
    0 = bg
    1 = A
    2 = B
    3 = C
    4 = D
    5 = TINB/TINC   (TIN-B 与 TIN-C 合并)
    6 = TIND
    7 = HH          (划痕，不展示)
    8 = XW          (纤维，不展示)
    9 = XQL         (镶嵌料，不展示)
    10 = HC         (灰尘，不展示)
    11 = SZ         (水渍，不展示)
"""

# ---------------------------------------------------------------------------
# 类别表
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "bg", "A", "B", "C", "D", "TINB/TINC", "TIND",
    "HH", "XW", "XQL", "HC", "SZ",
]
NUM_CLASSES_UNIFIED = len(CLASS_NAMES)  # 12

# 最终展示（业务目标）类别：A/B/C/D/TINB-C/TIN-D
INCLUSION_CLASSES = [1, 2, 3, 4, 5, 6]
INCLUSION_CLASS_NAMES = ["A", "B", "C", "D", "TINB/C", "TIND"]

# 条状族（Strip Head 监督空间）
STRIP_CLASSES = [1, 2, 3, 5, 6, 7, 8, 9]  # A/B/C/TINB-C/TIN-D/HH/XW/XQL
# 点状族（Point Head 监督空间）
POINT_CLASSES = [4, 10, 11]               # D/HC/SZ
# 噪声/干扰类（不展示，仅作显式困难负样本）
NOISE_CLASSES = [7, 8, 9, 10, 11]         # HH/XW/XQL/HC/SZ

NUM_STRIP_CLASSES = len(STRIP_CLASSES) + 1  # 9（含背景）
NUM_POINT_CLASSES = len(POINT_CLASSES) + 1  # 4（含背景）
NUM_GATE_CLASSES = 3                        # bg / strip / point

IGNORE_INDEX = 255

# ---------------------------------------------------------------------------
# unified 类别 -> head/gate 索引映射
# ---------------------------------------------------------------------------
STRIP_HEAD_MAP = {0: 0}
for _i, _c in enumerate(STRIP_CLASSES, start=1):
    STRIP_HEAD_MAP[_c] = _i

POINT_HEAD_MAP = {0: 0}
for _i, _c in enumerate(POINT_CLASSES, start=1):
    POINT_HEAD_MAP[_c] = _i

GATE_MAP = {0: 0}
for _c in STRIP_CLASSES:
    GATE_MAP[_c] = 1
for _c in POINT_CLASSES:
    GATE_MAP[_c] = 2

# 反向：head 索引 -> unified 类别（推理融合时把 head 概率映射回 12 类空间）
STRIP_HEAD_TO_UNIFIED = [0] + list(STRIP_CLASSES)  # [0,1,2,3,5,6,7,8,9]
POINT_HEAD_TO_UNIFIED = [0] + list(POINT_CLASSES)  # [0,4,10,11]

# 类别 -> gate 组（0=bg, 1=strip, 2=point）
CLS_TO_GROUP = {0: 0}
for _c in STRIP_CLASSES:
    CLS_TO_GROUP[_c] = 1
for _c in POINT_CLASSES:
    CLS_TO_GROUP[_c] = 2


def build_mapping_tensor(mapping, num_classes=NUM_CLASSES_UNIFIED, default=IGNORE_INDEX):
    """构造 [num_classes] 查询表：unified 值 -> 目标值；未覆盖项 -> default。"""
    import torch
    t = torch.full((num_classes,), default, dtype=torch.long)
    for k, v in mapping.items():
        t[k] = v
    return t


def gate_target_tensor():
    return build_mapping_tensor(GATE_MAP)


def strip_target_tensor():
    return build_mapping_tensor(STRIP_HEAD_MAP)


def point_target_tensor():
    return build_mapping_tensor(POINT_HEAD_MAP)
