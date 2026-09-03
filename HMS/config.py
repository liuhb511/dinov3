from dataclasses import dataclass

@dataclass
class HMSConfig:
    # dataset
    dataset_root: str = r"D:/lhb/datasets/HMS"
    image_size: int = 1024
    num_classes: int = 5
    red_class: int = 4

    # base DINOv3 model: keep existing models/ untouched
    backbone_name: str = "dinov3_model"
    freeze_backbone: bool = True

    # LAB-a* auxiliary branch
    lab_channels: int = 32
    lab_fusion_init: float = 0.5

    # train
    batch_size: int = 4
    num_workers: int = 4
    epochs: int = 100
    freeze_epochs: int = 60
    learning_rate: float = 1e-4
    unfreeze_lr: float = 1e-5
    weight_decay: float = 1e-4
    amp: bool = True
    grad_clip: float = 1.0
    seed: int = 42

    # loss
    ce_weight: float = 0.55
    dice_weight: float = 0.25
    red_tversky_weight: float = 0.20

    # class weight: [background, hd_w, hd_y, hd_t, red]
    # RED is intentionally larger to improve recall.
    class_weights: tuple = (1.0, 1.0, 1.0, 1.0, 2.0)

    # Tversky: beta > alpha penalizes RED false negatives more strongly.
    red_tversky_alpha: float = 0.30
    red_tversky_beta: float = 0.70
    red_tversky_gamma: float = 1.0

    save_dir: str = "./checkpoints/HMS_lab_2"
