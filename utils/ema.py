import copy

import torch
import torch.nn as nn


def create_ema_teacher(student: nn.Module) -> nn.Module:
    """
    创建与 Student 结构完全相同的 EMA Teacher。

    Teacher:
    1. 不进行反向传播
    2. 不加入 optimizer
    3. 只通过 EMA 更新参数
    """
    teacher = copy.deepcopy(student)
    teacher.eval()

    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    return teacher


@torch.no_grad()
def update_ema_teacher(
    teacher: nn.Module,
    student: nn.Module,
    momentum: float = 0.99,
) -> None:
    """
    EMA 更新：

    teacher = momentum * teacher
            + (1 - momentum) * student
    """

    if not 0.0 <= momentum < 1.0:
        raise ValueError(
            f"EMA momentum 必须位于 [0,1)，实际为 {momentum}"
        )

    teacher_parameters = dict(teacher.named_parameters())
    student_parameters = dict(student.named_parameters())

    if teacher_parameters.keys() != student_parameters.keys():
        raise RuntimeError(
            "Teacher 和 Student 的参数结构不一致"
        )

    for name, teacher_parameter in teacher_parameters.items():
        student_parameter = student_parameters[name]

        teacher_parameter.mul_(momentum).add_(
            student_parameter.detach(),
            alpha=1.0 - momentum,
        )

    # 同步 buffers。
    # 当前网络主要使用 GroupNorm，没有 BN 运行统计，
    # 但保留这部分可以兼容其他模块。
    teacher_buffers = dict(teacher.named_buffers())
    student_buffers = dict(student.named_buffers())

    for name, teacher_buffer in teacher_buffers.items():
        if name in student_buffers:
            teacher_buffer.copy_(student_buffers[name])