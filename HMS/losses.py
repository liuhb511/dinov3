import torch
import torch.nn as nn
import torch.nn.functional as F

def multiclass_soft_dice_loss(logits, target, num_classes, eps=1e-6):
    prob = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target.long(), num_classes=num_classes)
    one_hot = one_hot.permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    inter = (prob * one_hot).sum(dims)
    denom = prob.sum(dims) + one_hot.sum(dims)
    dice = (2.0 * inter + eps) / (denom + eps)

    # exclude background from mean
    if num_classes > 1:
        dice = dice[1:]
    return 1.0 - dice.mean()

def focal_tversky_red(
    logits,
    target,
    red_class=4,
    alpha=0.30,
    beta=0.70,
    gamma=1.0,
    eps=1e-6,
):
    """
    beta > alpha => false negatives cost more than false positives.
    This specifically targets incomplete RED masks / low RED recall.
    """
    p = torch.softmax(logits, dim=1)[:, red_class]
    y = (target == red_class).float()

    tp = (p * y).sum(dim=(1, 2))
    fp = (p * (1.0 - y)).sum(dim=(1, 2))
    fn = ((1.0 - p) * y).sum(dim=(1, 2))

    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return ((1.0 - tversky) ** gamma).mean()

class HMSRedLoss(nn.Module):
    """
    CE + multiclass Dice + RED-focused Focal-Tversky.

    The RED term is deliberately recall-oriented to reduce fragmented/missing RED areas.
    """
    def __init__(
        self,
        num_classes=5,
        red_class=4,
        class_weights=(1,1,1,1,2),
        ce_weight=0.55,
        dice_weight=0.25,
        red_tversky_weight=0.20,
        alpha=0.30,
        beta=0.70,
        gamma=1.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.red_class = int(red_class)
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.red_tversky_weight = float(red_tversky_weight)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32)
        )

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits, target.long(),
            weight=self.class_weights.to(logits.device)
        )
        dice = multiclass_soft_dice_loss(
            logits, target, self.num_classes
        )
        red_tv = focal_tversky_red(
            logits, target,
            red_class=self.red_class,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
        )
        total = (
            self.ce_weight * ce
            + self.dice_weight * dice
            + self.red_tversky_weight * red_tv
        )
        return total, {
            "ce": float(ce.detach()),
            "dice_loss": float(dice.detach()),
            "red_tversky": float(red_tv.detach()),
        }
