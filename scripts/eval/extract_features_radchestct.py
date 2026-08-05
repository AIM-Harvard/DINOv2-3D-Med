"""
Extract CLS token features from a PrimusM backbone for RadChestCT.

Usage:
    uv run python scripts/eval/extract_features_radchestct.py \\
        --checkpoint /path/to/last.ckpt \\
        --split train \\
        --output-dir log/evaluation/radchestct_features

    # Baseline (random init):
    uv run python scripts/eval/extract_features_radchestct.py \\
        --checkpoint random \\
        --split test \\
        --output-dir log/evaluation/radchestct_features
"""

import argparse
import os
import sys

# Bootstrap project root onto sys.path (same pattern as export_ckpt_to_nnunet.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pathlib import Path

import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Resize

DATASET_DIR = "/mnt/data1/datasets/RadChestCT"
INPUT_SHAPE = (160, 160, 160)   # cubic target after pad-then-resize

LABEL_CSV = {
    "train": "imgtrain_Abnormality_and_Location_Labels.csv",
    "val":   "imgvalid_Abnormality_and_Location_Labels.csv",
    "test":  "imgtest_Abnormality_and_Location_Labels.csv",
}


def load_label_csv(csv_path: Path) -> tuple[list[str], list[str], np.ndarray]:
    """
    Load label CSV without pandas.
    Returns:
        ids: list of NoteAcc_DEID strings
        pathologies: sorted list of 84 pathology names
        labels: float32 array [N, 84] (OR-aggregated per pathology)
    """
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    id_col = header.index('NoteAcc_DEID')
    loc_cols = [(i, c) for i, c in enumerate(header) if '*' in c]
    pathologies = sorted(set(c.split('*')[0] for _, c in loc_cols))
    path_to_col_indices = {
        p: [i for i, c in loc_cols if c.startswith(p + '*')]
        for p in pathologies
    }

    ids = [row[id_col] for row in rows]
    data = np.array([[float(row[i]) for i, _ in loc_cols] for row in rows], dtype=np.float32)
    # OR-aggregate: for each pathology, take max across its location columns
    loc_col_positions = [i for i, _ in loc_cols]
    labels = np.zeros((len(rows), len(pathologies)), dtype=np.float32)
    for j, path in enumerate(pathologies):
        col_indices_in_data = [
            k for k, (i, c) in enumerate(loc_cols) if c.startswith(path + '*')
        ]
        labels[:, j] = data[:, col_indices_in_data].max(axis=1)

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
    Loads RadChestCT .npz volumes and multi-label pathology targets.

    Preprocessing: HU clip [-1000, 1000] → scale [0, 1] → pad to cubic → Resize to INPUT_SHAPE
    Each item: (volume_tensor [1, 160, 160, 160] float32, label_tensor [84] float32)
    """

    def __init__(self, dataset_dir: str, split: str):
        self.dataset_dir = Path(dataset_dir)
        csv_name = LABEL_CSV[split]
        self.ids, self.pathologies, self.labels = load_label_csv(
            self.dataset_dir / csv_name
        )
        self.resize = Resize(spatial_size=INPUT_SHAPE, mode="trilinear")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        note_id = self.ids[idx]
        npz_path = self.dataset_dir / f"{note_id}.npz"
        ct = np.load(npz_path)['ct'].astype(np.float32)
        # HU clip + linear scale to [0, 1]
        ct = np.clip(ct, -1000.0, 1000.0)
        ct = (ct + 1000.0) / 2000.0
        ct = pad_to_cube(ct)                             # cubic, variable side length
        ct_tensor = torch.from_numpy(ct).unsqueeze(0)   # [1, S, S, S]
        ct_tensor = self.resize(ct_tensor)               # [1, 160, 160, 160]
        label = torch.from_numpy(self.labels[idx])       # [84]
        return ct_tensor, label


def load_backbone(checkpoint_path: str, seed: int = None):
    """
    Instantiate PrimusM and optionally load student backbone weights from .ckpt.
    Returns model in eval mode with frozen parameters.

    Args:
        checkpoint_path: path to .ckpt or literal "random" for random init.
        seed: if checkpoint_path=="random", fix torch seed for reproducibility.
    """
    from models.backbones.primus import Primus
    from utils.imports import import_module_from_path

    model = Primus(
        input_channels=1,
        embed_dim=864,
        patch_embed_size=(8, 8, 8),
        num_classes=1,
        eva_depth=16,
        eva_numheads=12,
        input_shape=INPUT_SHAPE,
        drop_path_rate=0.0,
        attn_drop_rate=0.0,
        patch_drop_rate=0.0,
        classification=True,
        init_values=0.1,
        scale_attn_inner=True,
    )

    if checkpoint_path == "random" and seed is not None:
        torch.manual_seed(seed)
        print(f"Random init with seed={seed}.")

    if checkpoint_path != "random":
        # Register project modules so torch.load can deserialize the Lightning ckpt
        project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        import_module_from_path("project", project_path)

        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except ModuleNotFoundError as e:
            print(f"Warning: Missing dependency ({e}), retrying with weights_only=True")
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        sd = ckpt["state_dict"]
        prefix = "model.student_backbone."
        # Skip RoPE buffers — computed on-the-fly, not needed for inference
        skip_suffixes = ('rope.periods',)
        filtered = {
            k[len(prefix):]: v for k, v in sd.items()
            if k.startswith(prefix) and not any(k.endswith(s) for s in skip_suffixes)
        }
        # Interpolate pos_embed if input resolution changed (e.g. 96³→160³)
        if 'vit.pos_embed' in filtered:
            ckpt_pe = filtered['vit.pos_embed']          # [1, 1+N_src, D]
            model_pe = model.vit.pos_embed               # [1, 1+N_tgt, D]
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

        missing, unexpected = model.load_state_dict(filtered, strict=False)
        # Only rope.periods should be missing (registered buffers, not in checkpoint)
        real_missing = [m for m in missing if 'rope.periods' not in m]
        if real_missing or unexpected:
            raise RuntimeError(
                f"Checkpoint load mismatch — missing: {real_missing[:5]}, unexpected: {unexpected[:5]}"
            )
        print(f"Loaded {len(filtered)} keys from student backbone: {checkpoint_path}")
    else:
        print("Using randomly initialized backbone (no checkpoint).")

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


def extract_features(
    model: torch.nn.Module,
    dataset: RadChestCTDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run all samples through backbone and return (features [N,864], labels [N,84])."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    model = model.to(device)

    all_features = []
    all_labels = []

    print(f"Extracting features from {len(dataset)} samples (batch_size={batch_size})...")
    with torch.no_grad():
        for i, (x, labels) in enumerate(loader):
            x = x.to(device)
            out = model(x)          # [B, N+1, 864]
            cls = out[:, 0, :].cpu()  # CLS token at position 0
            all_features.append(cls)
            all_labels.append(labels)
            if (i + 1) % 50 == 0:
                print(f"  {(i + 1) * batch_size}/{len(dataset)} samples processed")

    features = torch.cat(all_features, dim=0)  # [N, 864]
    labels = torch.cat(all_labels, dim=0)       # [N, 84]
    print(f"Done. Features: {features.shape}, Labels: {labels.shape}")
    return features, labels


def main():
    parser = argparse.ArgumentParser(description="Extract RadChestCT CLS token features.")
    parser.add_argument("--checkpoint", required=True,
                        help='Path to Lightning .ckpt, or literal "random" for random init baseline.')
    parser.add_argument("--dataset-dir", default=DATASET_DIR,
                        help=f"Path to RadChestCT root directory (default: {DATASET_DIR})")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"],
                        help="Dataset split to extract features from.")
    parser.add_argument("--output-dir", default="log/evaluation/radchestct_features",
                        help="Directory to save feature .pt files.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=None,
                        help='Random seed for random init baseline (only used when --checkpoint random).')
    args = parser.parse_args()

    # Validate inputs
    if args.checkpoint != "random" and not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.dataset_dir).is_dir():
        print(f"ERROR: Dataset directory not found: {args.dataset_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Derive model_id
    if args.checkpoint == "random":
        model_id = f"random_s{args.seed}" if args.seed is not None else "random"
    else:
        model_id = Path(args.checkpoint).stem

    output_path = output_dir / f"{args.split}_{model_id}_features.pt"
    print(f"Output: {output_path}")

    dataset = RadChestCTDataset(args.dataset_dir, args.split)
    print(f"Split: {args.split}, Samples: {len(dataset)}, Labels: {len(dataset.pathologies)}")

    model = load_backbone(args.checkpoint, seed=args.seed)
    features, labels = extract_features(model, dataset, args.batch_size, device)

    torch.save({
        "features": features,
        "labels": labels,
        "pathologies": dataset.pathologies,
        "model_id": model_id,
        "split": args.split,
        "checkpoint": args.checkpoint,
    }, output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
