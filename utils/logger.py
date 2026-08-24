"""
utils/logger.py

训练日志系统：
1. 保存配置 JSON
2. 每个 batch 实时记录训练 loss
3. 每个 epoch 记录训练和验证指标
4. 自动加载已有日志，支持断点续训
5. 自动绘制训练曲线，使用同名文件覆盖旧图
"""

import os
import json
import csv

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


class TrainLogger:
    def __init__(self, log_dir="./logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # 每个 epoch 的指标
        self.csv_path = os.path.join(self.log_dir, "metrics.csv")

        # 每个 batch 的实时训练日志
        self.batch_csv_path = os.path.join(self.log_dir, "batch_metrics.csv")

        # 配置文件
        self.config_path = os.path.join(self.log_dir, "config.json")

        # 固定曲线文件名，后续绘图时自动覆盖
        self.curve_path = os.path.join(self.log_dir, "training_curves.png")

        self.epoch_fieldnames = [
            "epoch",
            "train_loss",
            "val_loss",
            "val_iou",
            "val_dice",
            "lr",
        ]

        self.batch_fieldnames = [
            "epoch",
            "batch",
            "global_step",
            "batch_loss",
            "avg_loss",
            "lr",
        ]

        self.records = []

        # 加载之前已经保存的 epoch 日志，方便断点续训后继续画完整曲线
        self._load_existing_records()

    def _load_existing_records(self):
        """读取已有的 metrics.csv，恢复历史曲线数据。"""

        if not os.path.exists(self.csv_path):
            return

        if os.path.getsize(self.csv_path) == 0:
            return

        try:
            with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    # 跳过异常行或重复表头
                    try:
                        record = {
                            "epoch": int(row["epoch"]),
                            "train_loss": float(row["train_loss"]),
                            "val_loss": float(row["val_loss"]),
                            "val_iou": float(row["val_iou"]),
                            "val_dice": float(row["val_dice"]),
                            "lr": float(row["lr"]),
                        }
                    except (ValueError, TypeError, KeyError):
                        continue

                    self.records.append(record)

            if self.records:
                print(
                    f"[Logger] Loaded {len(self.records)} existing "
                    f"epoch records from {self.csv_path}"
                )

        except Exception as e:
            print(f"[Logger] Failed to load existing metrics: {e}")

    def save_config(self, cfg_dict):
        """保存训练配置。"""

        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(
                cfg_dict,
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        print(f"[Logger] Config saved to {self.config_path}")

    def log_batch(self, epoch, batch, global_step, batch_loss, avg_loss, lr):
        """
        实时记录每个 batch 的训练信息。

        文件每次以追加方式打开并立即关闭，因此程序中断时，
        已经完成的 batch 日志不会丢失。
        """

        record = {
            "epoch": epoch + 1,
            "batch": batch + 1,
            "global_step": global_step,
            "batch_loss": round(float(batch_loss), 6),
            "avg_loss": round(float(avg_loss), 6),
            "lr": float(lr),
        }

        file_exists = (
            os.path.exists(self.batch_csv_path)
            and os.path.getsize(self.batch_csv_path) > 0
        )

        with open(
            self.batch_csv_path,
            "a",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self.batch_fieldnames,
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(record)

            # 强制刷新 Python 缓冲区
            f.flush()

    def log_epoch(self, epoch, train_loss, val_loss, val_iou, val_dice, lr):
        """记录一个 epoch 的训练和验证指标。"""

        record = {
            "epoch": epoch + 1,
            "train_loss": round(float(train_loss), 6),
            "val_loss": round(float(val_loss), 6),
            "val_iou": round(float(val_iou), 6),
            "val_dice": round(float(val_dice), 6),
            "lr": float(lr),
        }

        # 避免断点续训时同一个 epoch 在内存中重复
        self.records = [
            old_record
            for old_record in self.records
            if old_record["epoch"] != record["epoch"]
        ]

        self.records.append(record)

        # 保证记录按照 epoch 排序
        self.records.sort(key=lambda item: item["epoch"])

        file_exists = (
            os.path.exists(self.csv_path)
            and os.path.getsize(self.csv_path) > 0
        )

        with open(
            self.csv_path,
            "a",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self.epoch_fieldnames,
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(record)
            f.flush()

        print(
            f"[Logger] Epoch {record['epoch']} logged | "
            f"train_loss={record['train_loss']:.6f} | "
            f"val_loss={record['val_loss']:.6f} | "
            f"iou={record['val_iou']:.6f} | "
            f"dice={record['val_dice']:.6f}"
        )

    def plot(self):
        """
        绘制训练曲线。

        保存路径固定为 training_curves.png，
        因此每次调用都会覆盖旧图。
        """

        if not self.records:
            print("[Logger] No epoch records available for plotting")
            return

        records = sorted(
            self.records,
            key=lambda item: item["epoch"],
        )

        epochs = [r["epoch"] for r in records]
        train_loss = [r["train_loss"] for r in records]
        val_loss = [r["val_loss"] for r in records]
        val_iou = [r["val_iou"] for r in records]
        val_dice = [r["val_dice"] for r in records]
        lr = [r["lr"] for r in records]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # ==========================
        # Train Loss 和 Val Loss
        # ==========================
        axes[0, 0].plot(
            epochs,
            train_loss,
            label="Train Loss",
            color="#1f77b4",
        )

        axes[0, 0].plot(
            epochs,
            val_loss,
            label="Val Loss",
            color="#ff7f0e",
        )

        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training & Validation Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # ==========================
        # IoU 和 Dice
        # ==========================
        axes[0, 1].plot(
            epochs,
            val_iou,
            label="Val IoU",
            color="#2ca02c",
        )

        axes[0, 1].plot(
            epochs,
            val_dice,
            label="Val Dice",
            color="#d62728",
        )

        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Score")
        axes[0, 1].set_title("Validation IoU & Dice")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # ==========================
        # 学习率
        # ==========================
        axes[1, 0].plot(
            epochs,
            lr,
            color="#9467bd",
        )

        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Learning Rate")
        axes[1, 0].set_title("Learning Rate Schedule")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 0].ticklabel_format(
            style="scientific",
            axis="y",
            scilimits=(0, 0),
        )

        # ==========================
        # Val Loss 和 Val Dice 双轴
        # ==========================
        ax1 = axes[1, 1]

        ax1.plot(
            epochs,
            val_loss,
            label="Val Loss",
            color="#ff7f0e",
        )

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel(
            "Val Loss",
            color="#ff7f0e",
        )

        ax1.tick_params(
            axis="y",
            labelcolor="#ff7f0e",
        )

        ax2 = ax1.twinx()

        ax2.plot(
            epochs,
            val_dice,
            label="Val Dice",
            color="#d62728",
        )

        ax2.set_ylabel(
            "Val Dice",
            color="#d62728",
        )

        ax2.tick_params(
            axis="y",
            labelcolor="#d62728",
        )

        ax1.set_title("Loss vs Dice")
        ax1.grid(True, alpha=0.3)

        fig.tight_layout()

        # 同一个路径，每次保存都会覆盖旧图片
        fig.savefig(
            self.curve_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(
            f"[Logger] Curves updated at epoch {epochs[-1]}: "
            f"{self.curve_path}"
        )