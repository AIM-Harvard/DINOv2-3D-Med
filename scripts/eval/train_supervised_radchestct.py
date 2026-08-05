"""
Fully supervised end-to-end Primus training on RadChestCT multi-label classification.

Trains a PrimusM backbone + linear classification head from scratch (or optionally
initializing from a DINOv2 pretrained checkpoint) on the 84-class RadChestCT dataset.

Architecture: PrimusM backbone (classification=True) → CLS token → Linear(864, 84)
Loss: BCEWithLogitsLoss (multi-label)
Optimizer: AdamW + cosine warmup schedule

Usage:
    # Random init (from scratch):
    uv run python scripts/eval/train_supervised_radchestct.py \\
        --run-name supervised_scratch

    # Finetune from DINOv2 pretrained weights:
    uv run python scripts/eval/train_supervised_radchestct.py \\
        --checkpoint /mnt/data1/suraj/.../last-v3.ckpt \\
        --run-name supervised_dino_finetune

    # Disable W&B:
    uv run python scripts/eval/train_supervised_radchestct.py --no-wandb
"""

import argparse
import csv
import os
import sys

# Bootstrap project root onto sys.path (same pattern as export_ckpt_to_nnunet.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision
from monai.transforms import Resize
from lightly.utils.scheduler import CosineWarmupScheduler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_DIR = "/mnt/data1/datasets/RadChestCT"
NUM_LABELS = 84
EMBED_DIM = 864
INPUT_SHAPE = (160, 160, 160)   # cubic target after pad-then-resize

LABEL_CSV = {
    "train": "imgtrain_Abnormality_and_Location_Labels.csv",
    "val":   "imgvalid_Abnormality_and_Location_Labels.csv",
    "test":  "imgtest_Abnormality_and_Location_Labels.csv",
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_label_csv(csv_path: Path):
    """
    Load RadChestCT label CSV without pandas.

    Returns:
        ids:         list of NoteAcc_DEID strings
        pathologies: sorted list of 84 pathology names
        labels:      float32 array [N, 84] (OR-aggregated per pathology)
    """
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    id_col = header.index('NoteAcc_DEID')
    loc_cols = [(i, c) for i, c in enumerate(header) if '*' in c]
    pathologies = sorted(set(c.split('*')[0] for _, c in loc_cols))

    data = np.array([[float(row[i]) for i, _ in loc_cols] for row in rows], dtype=np.float32)
    labels = np.zeros((len(rows), len(pathologies)), dtype=np.float32)
    for j, path in enumerate(pathologies):
        col_indices = [k for k, (_, c) in enumerate(loc_cols) if c.startswith(path + '*')]
        labels[:, j] = data[:, col_indices].max(axis=1)

    ids = [row[id_col] for row in rows]
    return ids, pathologies, labels


def pad_to_cube(ct: np.ndarray) -> np.ndarray:
    """
    Center-pad a 3D volume to cubic shape (all dims = max dim).

    Padding value is 0.0, which corresponds to HU=-1000 (air) after
    the linear scaling (x+1000)/2000. Applied after HU clip and scaling.
    """
    d, h, w = ct.shape
    target = max(d, h, w)
    pad = [(target - s) for s in (d, h, w)]
    padding = [(p // 2, p - p // 2) for p in pad]
    return np.pad(ct, padding, mode='constant', constant_values=0.0)


class RadChestCTDataset(Dataset):
    """
    Loads RadChestCT .npz CT volumes and 84-class multi-label targets.

    Preprocessing:
        HU clip [-1000, 1000] → scale [0, 1] → pad to cubic → Resize to INPUT_SHAPE
    Item: (volume_tensor [1, 160, 160, 160] float32, label_tensor [84] float32)
    """

    def __init__(self, dataset_dir: str, split: str):
        self.dataset_dir = Path(dataset_dir)
        self.ids, self.pathologies, self.labels = load_label_csv(
            self.dataset_dir / LABEL_CSV[split]
        )
        self.resize = Resize(spatial_size=INPUT_SHAPE, mode="trilinear")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        ct = np.load(self.dataset_dir / f"{self.ids[idx]}.npz")['ct'].astype(np.float32)
        ct = np.clip(ct, -1000.0, 1000.0)
        ct = (ct + 1000.0) / 2000.0
        ct = pad_to_cube(ct)                            # cubic, variable side length
        ct_tensor = torch.from_numpy(ct).unsqueeze(0)  # [1, S, S, S]
        ct_tensor = self.resize(ct_tensor)              # [1, 160, 160, 160]
        label = torch.from_numpy(self.labels[idx])     # [84]
        return ct_tensor, label


# ---------------------------------------------------------------------------
# Data module
# ---------------------------------------------------------------------------

class RadChestCTDataModule(LightningDataModule):
    def __init__(self, dataset_dir: str, batch_size: int = 2, num_workers: int = 4):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_ds = RadChestCTDataset(self.dataset_dir, "train")
        self.val_ds   = RadChestCTDataset(self.dataset_dir, "val")
        self.test_ds  = RadChestCTDataset(self.dataset_dir, "test")
        print(f"Dataset sizes — train: {len(self.train_ds)}, val: {len(self.val_ds)}, test: {len(self.test_ds)}")
        print(f"Labels: {len(self.train_ds.pathologies)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
            persistent_workers=(self.num_workers > 0),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=1, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=1, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )


# ---------------------------------------------------------------------------
# Backbone loading
# ---------------------------------------------------------------------------

def build_backbone(checkpoint_path=None):
    """
    Instantiate PrimusM backbone and optionally warm-start from a DINOv2 checkpoint.

    Args:
        checkpoint_path: path to Lightning .ckpt containing student backbone weights,
                         or None for random initialization.
    Returns:
        Primus model with requires_grad=True (trainable end-to-end).
    """
    from models.backbones.primus import Primus

    backbone = Primus(
        input_channels=1,
        embed_dim=EMBED_DIM,
        patch_embed_size=(8, 8, 8),
        num_classes=1,           # unused — CLS token is extracted before head
        eva_depth=16,
        eva_numheads=12,
        input_shape=INPUT_SHAPE,
        drop_path_rate=0.1,      # stochastic depth for regularisation during training
        attn_drop_rate=0.0,
        patch_drop_rate=0.0,
        classification=True,     # enables CLS token at position 0 of output
        init_values=0.1,
        scale_attn_inner=True,
    )

    if checkpoint_path is not None:
        from utils.imports import import_module_from_path
        project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        import_module_from_path("project", project_path)

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        prefix = "model.student_backbone."
        filtered = {
            k[len(prefix):]: v for k, v in sd.items()
            if k.startswith(prefix) and not k.endswith('rope.periods')
        }
        # Interpolate pos_embed if input resolution changed (e.g. 96³→160³)
        if 'vit.pos_embed' in filtered:
            ckpt_pe = filtered['vit.pos_embed']
            model_pe = backbone.vit.pos_embed
            if ckpt_pe.shape != model_pe.shape:
                N_src, D = ckpt_pe.shape[1] - 1, ckpt_pe.shape[2]
                N_tgt = model_pe.shape[1] - 1
                s_src = round(N_src ** (1 / 3))
                s_tgt = round(N_tgt ** (1 / 3))
                print(f"Interpolating pos_embed: {s_src}³→{s_tgt}³ ({N_src}→{N_tgt} patches)")
                cls_tok  = ckpt_pe[:, :1, :]
                patch_pe = ckpt_pe[:, 1:, :].permute(0, 2, 1).view(1, D, s_src, s_src, s_src)
                patch_pe = F.interpolate(patch_pe, size=(s_tgt, s_tgt, s_tgt),
                                         mode='trilinear', align_corners=False)
                patch_pe = patch_pe.view(1, D, -1).permute(0, 2, 1)
                filtered['vit.pos_embed'] = torch.cat([cls_tok, patch_pe], dim=1)

        missing, unexpected = backbone.load_state_dict(filtered, strict=False)
        real_missing = [m for m in missing if 'rope.periods' not in m]
        if real_missing or unexpected:
            raise RuntimeError(
                f"Checkpoint load mismatch — missing: {real_missing[:5]}, unexpected: {unexpected[:5]}"
            )
        print(f"Loaded {len(filtered)} SSL pretrained keys from {checkpoint_path}")
    else:
        print("Backbone initialized randomly (no pretrained checkpoint).")

    return backbone


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class SupervisedPrimusModule(LightningModule):
    """
    End-to-end supervised PrimusM classifier for RadChestCT.

    Pipeline:
        x [B, 1, 96, 96, 96]
        → backbone(x) [B, N+1, 864]
        → CLS token [:, 0, :] [B, 864]
        → Linear head [B, 84]
        → BCEWithLogitsLoss

    Metrics (val and test): macro AUC-ROC, macro mAP (torchmetrics)
    """

    def __init__(
        self,
        num_labels: int = NUM_LABELS,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 5,
        max_epochs: int = 50,
        checkpoint_path: str = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.backbone = build_backbone(checkpoint_path)
        self.head = nn.Linear(EMBED_DIM, num_labels)
        self.criterion = nn.BCEWithLogitsLoss()

        self.val_auroc  = MultilabelAUROC(num_labels=num_labels, average="macro")
        self.val_ap     = MultilabelAveragePrecision(num_labels=num_labels, average="macro")
        self.test_auroc = MultilabelAUROC(num_labels=num_labels, average="macro")
        self.test_ap    = MultilabelAveragePrecision(num_labels=num_labels, average="macro")

    def forward(self, x):
        out = self.backbone(x)   # [B, N+1, 864]
        cls = out[:, 0, :]       # [B, 864]  — CLS token at position 0
        return self.head(cls)    # [B, num_labels]

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        probs = torch.sigmoid(logits)
        self.val_auroc.update(probs, y.long())
        self.val_ap.update(probs, y.long())
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        auc = self.val_auroc.compute()
        ap  = self.val_ap.compute()
        self.log("val/macro_auc", auc, prog_bar=True)
        self.log("val/map", ap, prog_bar=True)
        self.val_auroc.reset()
        self.val_ap.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probs = torch.sigmoid(logits)
        self.test_auroc.update(probs, y.long())
        self.test_ap.update(probs, y.long())

    def on_test_epoch_end(self):
        auc = self.test_auroc.compute()
        ap  = self.test_ap.compute()
        self.log("test/macro_auc", auc)
        self.log("test/map", ap)
        print(f"\n[Test] macro AUC: {auc:.4f}   mAP: {ap:.4f}")
        self.test_auroc.reset()
        self.test_ap.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        total_steps = self.trainer.estimated_stepping_batches
        steps_per_epoch = max(total_steps // max(self.max_epochs, 1), 1)
        warmup_steps = self.warmup_epochs * steps_per_epoch
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_epochs=warmup_steps,
            max_epochs=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Supervised end-to-end PrimusM training on RadChestCT."
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to DINOv2 Lightning .ckpt for backbone weight init. "
             "Default: random initialization.",
    )
    parser.add_argument("--dataset-dir", default=DATASET_DIR,
                        help=f"RadChestCT root directory (default: {DATASET_DIR})")
    parser.add_argument("--run-name", default="supervised_radchestct",
                        help="W&B run name and output subdirectory label")
    parser.add_argument("--output-dir", default="log/supervised/radchestct",
                        help="Root directory for checkpoints and logs")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Number of linear LR warmup epochs before cosine decay")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--devices", type=int, default=1,
                        help="Number of GPUs to use")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging (use CSV logger instead)")
    parser.add_argument("--test-only", default=None, metavar="CKPT",
                        help="Skip training; evaluate this checkpoint on the test split")
    args = parser.parse_args()

    # Validate inputs
    if args.checkpoint is not None and not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.dataset_dir).is_dir():
        print(f"ERROR: Dataset directory not found: {args.dataset_dir}", file=sys.stderr)
        sys.exit(1)
    if args.test_only is not None and not Path(args.test_only).exists():
        print(f"ERROR: --test-only checkpoint not found: {args.test_only}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    datamodule = RadChestCTDataModule(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Model
    model = SupervisedPrimusModule(
        num_labels=NUM_LABELS,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        checkpoint_path=args.checkpoint,
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="best-{epoch:02d}-auc{val/macro_auc:.4f}",
            monitor="val/macro_auc",
            mode="max",
            save_last=True,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Logger
    if args.no_wandb:
        logger = pl.loggers.CSVLogger(save_dir=str(output_dir), name="csv_logs")
    else:
        logger = WandbLogger(
            project="dinov2_3d",
            name=args.run_name,
            save_dir=str(output_dir),
            tags=["supervised", "radchestct"],
        )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=args.devices,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        precision="16-mixed",
        gradient_clip_val=1.0,
        enable_progress_bar=True,
    )

    if args.test_only is not None:
        print(f"Test-only mode: loading {args.test_only}")
        trainer.test(model, datamodule=datamodule, ckpt_path=args.test_only)
    else:
        trainer.fit(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
