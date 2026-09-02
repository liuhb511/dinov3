import torch

class SegmentationMeter:
    def __init__(self, num_classes=5, red_class=4, device="cpu"):
        self.num_classes = int(num_classes)
        self.red_class = int(red_class)
        self.cm = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.long, device=device
        )

    @torch.no_grad()
    def update(self, pred, target):
        pred = pred.reshape(-1).long()
        target = target.reshape(-1).long()
        valid = (
            (target >= 0) & (target < self.num_classes)
            & (pred >= 0) & (pred < self.num_classes)
        )
        k = target[valid] * self.num_classes + pred[valid]
        bins = torch.bincount(
            k, minlength=self.num_classes ** 2
        )
        self.cm += bins.reshape(self.num_classes, self.num_classes)

    def compute(self):
        cm = self.cm.float()
        tp = cm.diag()
        gt = cm.sum(dim=1)
        pd = cm.sum(dim=0)
        union = gt + pd - tp

        iou = tp / union.clamp_min(1)
        dice = 2 * tp / (gt + pd).clamp_min(1)
        recall = tp / gt.clamp_min(1)
        precision = tp / pd.clamp_min(1)

        fg_iou = iou[1:].mean().item() if self.num_classes > 1 else iou.mean().item()
        fg_dice = dice[1:].mean().item() if self.num_classes > 1 else dice.mean().item()
        r = self.red_class

        return {
            "mIoU_fg": fg_iou,
            "mDice_fg": fg_dice,
            "red_iou": iou[r].item(),
            "red_dice": dice[r].item(),
            "red_recall": recall[r].item(),
            "red_precision": precision[r].item(),
            "class_iou": iou.cpu().tolist(),
            "class_dice": dice.cpu().tolist(),
        }
