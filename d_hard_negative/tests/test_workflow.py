import json
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from lib.common import ROOT, exact_region, map_gt, output_path, read_csv, read_image, save_image, write_csv
from lib.dataset import DDataset
from lib.inference import predict
from lib.loss import DLoss


class WorkflowTests(unittest.TestCase):
    def test_training_and_audit_entrypoints(self):
        def module(name):
            spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name+".py"))
            result = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(result)
            return result

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Identity()
                self.conv = torch.nn.Conv2d(3, 3, 1)

            def forward(self, image):
                values = self.conv(image)
                return values[:, :2], values[:, 2:]

        with tempfile.TemporaryDirectory(dir=ROOT, prefix="test_") as tmp:
            root = Path(tmp)
            rows = []
            for split in ("train", "audit"):
                image = np.zeros((8, 8, 3), np.uint8)
                gt = np.zeros((8, 8), np.uint8)
                gt[3, 3] = 1
                negative = np.zeros_like(gt)
                negative[6, 6] = 1
                paths = {}
                for name, value in (("image", image), ("gt", gt), ("negative", negative), ("ignore", np.zeros_like(gt))):
                    path = root / split / (name+".png")
                    save_image(path, value)
                    paths[name+"_path"] = str(path)
                rows.append(dict(image_id=split, split=split, group_id=split, has_negative=1, **paths))
            supervision = "../../"+root.name+"/supervision"
            write_csv(root / "supervision/manifest.csv", rows)
            component = root / "component.png"
            save_image(component, np.ones((1, 1), np.uint8))
            write_csv(root / "supervision/audit_targets.csv", [dict(image_id="audit", candidate_id="confirmed_d", x=3, y=3, w=1, h=1, component_mask_path=str(component), review_label="1", positive_difficulty="weak"),
                                                              dict(image_id="audit", candidate_id="negative", x=6, y=6, w=1, h=1, component_mask_path=str(component), review_label="0", positive_difficulty="")])
            cfg = dict(supervision_id=supervision, training_run="../"+root.name+"/training",
                       audit_run="../"+root.name+"/audit", image_size=8, window_size=8, device="cpu",
                       freeze_backbone=True, epochs=1, batch_size=1, checkpoint="unused.pth", d_threshold=.5)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(cfg))
            trainer = module("train_d_hard_negatives")
            fake_loader = lambda *a, **kw: (Tiny(), torch.device("cpu"), "synthetic")
            with patch.object(trainer, "load_model", fake_loader), patch.object(sys, "argv", ["train", "--config", str(config_path), "--max-steps", "1"]):
                trainer.main()
            self.assertTrue((root / "training/checkpoints/epoch_001.pth").exists())
            log = read_csv(root / "training/logs/epochs.csv")
            self.assertEqual(float(log[0]["negative_pixels"]), 1)
            evaluator = module("evaluate_d_audit")
            with patch.object(evaluator, "load_model", fake_loader), patch.object(sys, "argv", ["audit", "--config", str(config_path), "--checkpoint", "unused_new.pth"]):
                evaluator.main()
            summary = json.loads((root / "audit/evaluation/summary.json").read_text())
            self.assertEqual(summary["baseline"]["positive_review"]["targets"], 1)
            self.assertEqual(summary["new"]["weak_positive"]["targets"], 1)
            self.assertEqual(summary["new"]["positive_gt_unverified"]["targets"], 0)

    def test_output_guard_and_mapping(self):
        with self.assertRaises(ValueError):
            output_path("../escape.png")
        with self.assertRaises(ValueError):
            map_gt(np.array([[4]], np.uint8), {"0": 0, "1": 1})
        mapped = map_gt(np.array([[0, 4, 6, 255]], np.uint8), {0: 0, 4: 1, 6: 0, 255: 255})
        np.testing.assert_array_equal(mapped, [[0, 1, 0, 255]])

    def test_auxiliary_gradient_and_ignore(self):
        logits = torch.zeros(2, 2, 5, 5, requires_grad=True)
        gt = torch.zeros(2, 5, 5, dtype=torch.long)
        gt[:, 0, 0] = 255
        negative = torch.zeros_like(gt, dtype=torch.bool)
        negative[0, 2, 2] = True
        negative[0, 0, 0] = True
        loss, parts = DLoss(dict(ce_weight=0, dice_weight=0, boundary_weight=0, hard_negative_weight=1))(logits, None, gt, negative)
        loss.backward()
        self.assertGreater(logits.grad[0, 1, 2, 2].item(), 0)
        self.assertEqual(torch.count_nonzero(logits.grad).item(), 2)
        self.assertEqual(int(parts["negative_pixels"]), 1)
        logits2 = torch.zeros_like(logits, requires_grad=True)
        loss2, _ = DLoss({})(logits2, None, torch.full_like(gt, 255), negative)
        self.assertEqual(float(loss2), 0)
        loss2.backward()
        self.assertEqual(torch.count_nonzero(logits2.grad).item(), 0)

    def test_boundary_ignore_and_real_update(self):
        model = torch.nn.Conv2d(3, 3, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        image = torch.randn(1, 3, 8, 8)
        output = model(image)
        output.retain_grad()
        gt = torch.zeros(1, 8, 8, dtype=torch.long)
        gt[:, 3, 3] = 1
        gt[:, 0:2, 0:2] = 255
        negative = torch.zeros_like(gt, dtype=torch.bool)
        negative[:, 6, 6] = True
        before = model.weight.detach().clone()
        loss, _ = DLoss({})(output[:, :2], output[:, 2:], gt, negative)
        loss.backward()
        self.assertEqual(torch.count_nonzero(output.grad[:, :2, :2, :2]).item(), 0)
        self.assertEqual(torch.count_nonzero(output.grad[:, 2, :3, :3]).item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.weight))

    def test_geometry_and_mask_build(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix="test_") as tmp:
            root = Path(tmp)
            image = np.zeros((8, 12, 3), np.uint8)
            gt = np.zeros((8, 12), np.uint8)
            gt[3, 4] = 1
            negative = np.zeros_like(gt)
            negative[5, 8] = 1
            image[negative > 0] = 255
            image[gt > 0] = 255
            paths = {}
            for name, value in (("image", image), ("gt", gt), ("negative", negative), ("ignore", np.zeros_like(gt))):
                paths[name] = root / (name+".png")
                save_image(paths[name], value)
            row = dict(image_id="image", image_path=str(paths["image"]), gt_path=str(paths["gt"]),
                       negative_path=str(paths["negative"]), ignore_path=str(paths["ignore"]))
            random.seed(42)
            tensor, target, hn, _ = DDataset([row], size=12, augment=True)[0]
            self.assertEqual(int(hn.sum()), 1)
            self.assertEqual(int((target == 1).sum()), 1)
            self.assertFalse((hn & (target == 1)).any())
            self.assertGreater(float(tensor[:, hn].mean()), float(tensor[:, target == 0].mean()))
            component = root / "component.png"
            save_image(component, np.ones((1, 1), np.uint8)*255)
            candidates = []
            for cid, x, y, label in (("negative", 8, 5, "0"), ("conflict", 4, 3, "0"), ("uncertain", 2, 2, "3"), ("positive", 6, 4, "1")):
                candidates.append(dict(candidate_id=cid, image_id="image", x=x, y=y, w=1, h=1,
                                       component_mask_path=str(component), review_label=label, positive_mask_path=""))
            review = root / "review.csv"
            write_csv(review, candidates)
            split = root / "split.csv"
            audit_image = root / "audit_image.png"
            save_image(audit_image, image)
            write_csv(split, [dict(image_id="image", image_path=str(paths["image"]), gt_mask_path=str(paths["gt"]), group_id="g1", split="train"),
                              dict(image_id="audit", image_path=str(audit_image), gt_mask_path=str(paths["gt"]), group_id="g2", split="audit")])
            cfg = dict(review_csv=str(review), split_csv=str(split), supervision_id=root.name,
                       gt_mapping={"0": 0, "1": 1})
            config_path = root / "config.json"
            config_path.write_text(json.dumps(cfg))
            # Build below a temporary task-local subtree, cleaned with TemporaryDirectory.
            cfg["supervision_id"] = "../../"+root.name+"/built"
            config_path.write_text(json.dumps(cfg))
            subprocess.run([sys.executable, str(ROOT / "scripts/build_d_supervision_masks.py"), "--config", str(config_path)], check=True, capture_output=True)
            built = root / "built"
            hn = read_image(built / "train/negative/image.png", True)
            ig = read_image(built / "train/ignore/image.png", True)
            self.assertEqual(int(hn.sum()), 1)
            self.assertEqual(int(ig[3, 4]), 1)
            self.assertEqual(int(ig[4, 6]), 1)
            self.assertEqual(len(read_csv(built / "conflicts.csv")), 1)
            region = exact_region(candidates[0], (8, 12))
            self.assertTrue(region[5, 8])

    def test_inference_modes(self):
        class Dummy(torch.nn.Module):
            def forward(self, x):
                b, _, h, w = x.shape
                logits = torch.zeros(b, 2, h, w, device=x.device)
                logits[:, 1] = torch.arange(w, device=x.device)[None, None, :]
                return logits, None
        image = np.zeros((6, 9, 3), np.uint8)
        cfg = dict(window_size=4, fp16=False, overlap_mode="overwrite")
        over = predict(Dummy(), image, torch.device("cpu"), cfg)
        avg = predict(Dummy(), image, torch.device("cpu"), dict(cfg, overlap_mode="average"))
        self.assertEqual(over.shape, (6, 9))
        self.assertTrue(np.isfinite(avg).all())
        self.assertFalse(np.array_equal(over, avg))

    def test_mining_micro_points(self):
        spec = importlib.util.spec_from_file_location("miner", ROOT / "scripts/mine_d_hard_negatives.py")
        miner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(miner)

        class Points(torch.nn.Module):
            def forward(self, image):
                logits = torch.zeros(image.shape[0], 2, 8, 8)
                logits[:, 1] = -8
                logits[:, 1, 3, 3] = 8
                logits[:, 1, 6, 6] = 8
                return logits, None

        with tempfile.TemporaryDirectory(dir=ROOT, prefix="test_") as tmp:
            root = Path(tmp)
            save_image(root / "images/source.png", np.zeros((8, 8, 3), np.uint8))
            save_image(root / "masks/source.png", np.zeros((8, 8), np.uint8))
            cfg = dict(run_id="../../"+root.name+"/mined", image_dir=str(root / "images"), mask_dir=str(root / "masks"),
                       gt_mapping={"0": 0, "1": 1}, d_threshold=.5, window_size=8, fp16=False, d_margin=0)
            path = root / "config.json"
            path.write_text(json.dumps(cfg))
            with patch.object(miner, "load_model", lambda cfg: (Points(), torch.device("cpu"), "synthetic")), patch.object(sys, "argv", ["mine", "--config", str(path)]):
                miner.main()
            rows = read_csv(root / "mined/candidates.csv")
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(int(row["area_px"]) == 1 for row in rows))
            self.assertEqual(int(exact_region(rows[0], (8, 8)).sum()), 1)


if __name__ == "__main__":
    unittest.main()
