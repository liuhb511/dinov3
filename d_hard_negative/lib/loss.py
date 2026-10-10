import torch
import torch.nn as nn
import torch.nn.functional as F


class DLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ce_weight = float(cfg.get("ce_weight", .7))
        self.dice_weight = float(cfg.get("dice_weight", .3))
        self.boundary_weight = float(cfg.get("boundary_weight", .2))
        self.hn_weight = float(cfg.get("hard_negative_weight", .5))
        if min(self.ce_weight, self.dice_weight, self.boundary_weight, self.hn_weight) < 0:
            raise ValueError("Loss weights must be nonnegative")

    def forward(self, logits, boundary, gt, negative):
        logits = logits.float()
        if logits.shape[1] != 2 or logits.shape[2:] != gt.shape[1:]:
            raise ValueError("Expected aligned two-class logits")
        valid = gt != 255
        safe_gt = gt.masked_fill(~valid, 0)
        ce_map = F.cross_entropy(logits, safe_gt, reduction="none")
        zero = logits.sum()*0
        ce = ce_map[valid].mean() if valid.any() else zero
        pd = logits.softmax(1)[:, 1]
        positive = (gt == 1).float()
        pv = pd*valid
        dice = 1-(2*(pv*positive).sum()+1)/(pv.sum()+positive.sum()+1) if valid.any() else zero
        region = negative.bool() & valid
        if (region & (gt == 1)).any():
            raise ValueError("Negative region overlaps positive D")
        count = region.sum((1, 2))
        background_ce = F.cross_entropy(logits, torch.zeros_like(gt), reduction="none")
        per_image = (background_ce*region).sum((1, 2))/count.clamp_min(1)
        hn = per_image[count > 0].mean() if (count > 0).any() else zero
        edge_loss = zero
        if boundary is not None and self.boundary_weight:
            # Match original Laplacian edge definition, but mask neighborhoods touching ignore.
            kernel = logits.new_tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])[None, None]
            edge = (F.conv2d(positive[:, None], kernel, padding=1).abs() > 0).float()
            boundary_valid = F.avg_pool2d(valid[:, None].float(), 3, 1, 1, count_include_pad=True) == 1
            if boundary.shape[2:] != gt.shape[1:]:
                boundary = F.interpolate(boundary.float(), size=gt.shape[1:], mode="bilinear", align_corners=False)
            if boundary_valid.any():
                edge_loss = F.binary_cross_entropy_with_logits(boundary.float(), edge, reduction="none")[boundary_valid].mean()
        total = self.ce_weight*ce+self.dice_weight*dice+self.boundary_weight*edge_loss+self.hn_weight*hn
        return total, dict(ce=ce.detach(), dice=dice.detach(), boundary=edge_loss.detach(),
                          hard_negative=hn.detach(), negative_pixels=region.sum().detach(),
                          negative_images=(count > 0).sum().detach())
