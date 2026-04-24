"""
Fit a multi-label linear probe on extracted RadChestCT CLS token features
and evaluate with macro AUC-ROC and mAP.

Usage:
    uv run python scripts/eval/linear_probe_radchestct.py \\
        --train-features log/evaluation/radchestct_features/train_last-v3_features.pt \\
        --test-features  log/evaluation/radchestct_features/test_last-v3_features.pt \\
        --model-label "SSL (DINOv2 PrimusM)"

    # Then baseline:
    uv run python scripts/eval/linear_probe_radchestct.py \\
        --train-features log/evaluation/radchestct_features/train_random_features.pt \\
        --test-features  log/evaluation/radchestct_features/test_random_features.pt \\
        --model-label "Baseline (random init)"

    # Compare SSL vs baseline (pass both result JSON files):
    uv run python scripts/eval/linear_probe_radchestct.py --compare \\
        --ssl-results log/evaluation/radchestct_features/results_last-v3.json \\
        --baseline-results log/evaluation/radchestct_features/results_random.json \\
        --output-dir log/evaluation
"""

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision


def load_features(path: str) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data


def fit_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    lr: float = 0.01,
    max_iter: int = 100,
) -> nn.Linear:
    """
    Fit a multi-label logistic regression using L-BFGS.
    features: [N, D], labels: [N, num_labels] float32
    """
    num_labels = labels.shape[1]
    probe = nn.Linear(features.shape[1], num_labels).to(device)
    features = features.to(device)
    labels = labels.to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.LBFGS(
        probe.parameters(),
        lr=lr,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(probe(features), labels)
        loss.backward()
        return loss

    print(f"Fitting linear probe ({features.shape[0]} samples, {num_labels} labels)...")
    optimizer.step(closure)
    with torch.no_grad():
        final_loss = loss_fn(probe(features), labels).item()
    print(f"Probe fitted. Final train BCE loss: {final_loss:.4f}")
    return probe.cpu()


def evaluate(
    probe: nn.Linear,
    features: torch.Tensor,
    labels: torch.Tensor,
    num_labels: int,
    device: torch.device,
) -> dict:
    """Evaluate probe on given features/labels. Returns macro_auc and map."""
    probe = probe.to(device)
    features = features.to(device)
    labels_int = labels.long().to(device)

    auroc_metric = MultilabelAUROC(num_labels=num_labels, average="macro").to(device)
    ap_metric = MultilabelAveragePrecision(num_labels=num_labels, average="macro").to(device)

    with torch.no_grad():
        logits = probe(features)
        probs = torch.sigmoid(logits)

    macro_auc = auroc_metric(probs, labels_int).item()
    map_score = ap_metric(probs, labels_int).item()
    return {"macro_auc": macro_auc, "map": map_score}


def run_probe(args) -> dict:
    """Full train+eval pipeline for a single model."""
    device = torch.device(args.device)

    print(f"\n=== {args.model_label} ===")
    print(f"Loading train features: {args.train_features}")
    train_data = load_features(args.train_features)
    train_features = train_data["features"]
    train_labels = train_data["labels"]
    pathologies = train_data["pathologies"]
    num_labels = len(pathologies)

    print(f"Loading test features: {args.test_features}")
    test_data = load_features(args.test_features)
    test_features = test_data["features"]
    test_labels = test_data["labels"]

    print(f"Train: {train_features.shape}, Test: {test_features.shape}, Labels: {num_labels}")

    probe = fit_probe(train_features, train_labels, device, lr=args.lr, max_iter=args.max_iter)
    metrics = evaluate(probe, test_features, test_labels, num_labels, device)

    print(f"  Macro AUC-ROC: {metrics['macro_auc']:.4f}")
    print(f"  mAP:           {metrics['map']:.4f}")

    result = {
        "model_label": args.model_label,
        "checkpoint": train_data.get("checkpoint", "unknown"),
        "model_id": train_data.get("model_id", "unknown"),
        "train_samples": train_features.shape[0],
        "test_samples": test_features.shape[0],
        "num_labels": num_labels,
        "macro_auc": metrics["macro_auc"],
        "map": metrics["map"],
    }

    # Save individual result JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_id = train_data.get("model_id", "unknown")
    json_path = output_dir / f"results_{model_id}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {json_path}")
    return result


def compare_and_write_report(ssl_path: str, baseline_paths: list, output_dir: str):
    """
    Load SSL result JSON and one or more baseline JSONs, write markdown report.

    With multiple baselines the report shows mean ± std and [min, max] range.
    """
    import math

    with open(ssl_path) as f:
        ssl = json.load(f)

    baselines = []
    for p in baseline_paths:
        with open(p) as f:
            baselines.append(json.load(f))

    base_aucs = [b["macro_auc"] for b in baselines]
    base_maps  = [b["map"] for b in baselines]

    def _stats(vals):
        n = len(vals)
        mean = sum(vals) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / n) if n > 1 else 0.0
        return mean, std, min(vals), max(vals)

    b_auc_mean, b_auc_std, b_auc_min, b_auc_max = _stats(base_aucs)
    b_map_mean, b_map_std, b_map_min, b_map_max = _stats(base_maps)

    delta_auc = ssl["macro_auc"] - b_auc_mean
    delta_map = ssl["map"] - b_map_mean

    print("\n=== Delta Comparison ===")
    print(f"  SSL macro-AUC:           {ssl['macro_auc']:.4f}")
    print(f"  Baseline macro-AUC mean: {b_auc_mean:.4f} ± {b_auc_std:.4f}  [{b_auc_min:.4f}, {b_auc_max:.4f}]")
    print(f"  Delta AUC (vs mean):     {delta_auc:+.4f}")
    print(f"  SSL mAP:                 {ssl['map']:.4f}")
    print(f"  Baseline mAP mean:       {b_map_mean:.4f} ± {b_map_std:.4f}  [{b_map_min:.4f}, {b_map_max:.4f}]")
    print(f"  Delta mAP (vs mean):     {delta_map:+.4f}")

    # Build per-seed baseline rows for the table
    baseline_rows = "\n".join(
        f"| {b['model_label']} | {b['macro_auc']:.4f} | {b['map']:.4f} |"
        for b in baselines
    )
    n_seeds = len(baselines)
    baseline_summary = (
        f"| Baseline mean (n={n_seeds}) | {b_auc_mean:.4f} ± {b_auc_std:.4f} [{b_auc_min:.4f}–{b_auc_max:.4f}] | "
        f"{b_map_mean:.4f} ± {b_map_std:.4f} [{b_map_min:.4f}–{b_map_max:.4f}] |"
    )

    report = f"""# Evaluation Results — RadChestCT Classification Probe

**Date:** {date.today().isoformat()}
**SSL Checkpoint:** {ssl['checkpoint']}
**Dataset:** /mnt/data1/datasets/RadChestCT
**Num labels:** {ssl['num_labels']} (OR-aggregated pathology types)
**Linear probe:** nn.Linear(864, {ssl['num_labels']}) + L-BFGS, lr=0.01, max_iter=100
**Train samples:** {ssl['train_samples']} | **Test samples:** {ssl['test_samples']}

## Results

| Model | Macro AUC-ROC | mAP |
|-------|---------------|-----|
| {ssl['model_label']} | {ssl['macro_auc']:.4f} | {ssl['map']:.4f} |
{baseline_rows}
{baseline_summary}
| **Delta (SSL − Baseline mean)** | **{delta_auc:+.4f}** | **{delta_map:+.4f}** |

## Notes

- Baseline: {n_seeds} independent random seeds — range shows seed sensitivity
- Label aggregation: OR across all location columns per pathology type
- Evaluation split: test ({ssl['test_samples']} samples)
- Backbone: PrimusM (embed_dim=864, eva_depth=16, input_shape=160³, patch_size=8³)
- CLS token: output[:, 0, :] from frozen backbone (no gradient)
- CT preprocessing: HU clip [-1000, 1000] → scale [0,1] → pad to cube → MONAI Resize to [160,160,160]
"""

    out_path = Path(output_dir) / "radchestct_results.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"\nReport written to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Fit and evaluate RadChestCT linear probe.")
    subparsers = parser.add_subparsers(dest="mode")

    # Default mode: run single probe
    parser.add_argument("--train-features", help="Path to train features .pt file.")
    parser.add_argument("--test-features", help="Path to test features .pt file.")
    parser.add_argument("--model-label", default="SSL (DINOv2 PrimusM)",
                        help="Human-readable label for this model in results.")
    parser.add_argument("--output-dir", default="log/evaluation/radchestct_features",
                        help="Directory to write result JSON.")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Compare mode
    parser.add_argument("--compare", action="store_true",
                        help="Compare SSL vs baseline from saved JSON results.")
    parser.add_argument("--ssl-results", help="Path to SSL result JSON (for --compare).")
    parser.add_argument("--baseline-results", nargs="+",
                        help="Path(s) to baseline result JSON(s) (for --compare). "
                             "Multiple paths show mean ± std and range.")

    args = parser.parse_args()

    if args.compare:
        if not args.ssl_results or not args.baseline_results:
            print("ERROR: --compare requires --ssl-results and --baseline-results", file=sys.stderr)
            sys.exit(1)
        compare_and_write_report(
            args.ssl_results, args.baseline_results,
            output_dir=args.output_dir,
        )

    else:
        if not args.train_features or not args.test_features:
            print("ERROR: Provide --train-features and --test-features (or use --compare).",
                  file=sys.stderr)
            sys.exit(1)
        run_probe(args)


if __name__ == "__main__":
    main()
