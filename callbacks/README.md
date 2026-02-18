# Callbacks

This directory contains PyTorch Lightning callbacks for DINOv2 3D training.

## GramMatrixCallback

A callback that logs a gram matrix (similarity matrix) of features to Weights & Biases during training.

### What is a Gram Matrix?

The gram matrix shows the cosine similarity between normalized features for all samples in a batch. Each cell (i, j) in the matrix represents the cosine similarity between sample i and sample j, where:
- Value of 1 (bright) = features are identical/very similar
- Value of 0 (medium) = features are orthogonal/unrelated
- Value of -1 (dark) = features are opposite

This visualization helps understand:
- How diverse the learned representations are
- Whether the model is learning meaningful distinctions between samples
- If there are duplicates or very similar samples in the batch

### Usage

The callback is already configured in `configs/train.yaml`:

```yaml
callbacks:
  - _target_: project.callbacks.GramMatrixCallback
    log_every_n_steps: 100      # Log every 100 training steps
    feature_key: "teacher_cls_token"  # Which features to use
    max_samples: 128            # Max samples to include (prevents OOM)
    saturation_offdiag_threshold: 0.995
    saturation_offdiag_std_threshold: 0.01
    saturation_sample_var_threshold: 1.0e-6
    auto_fallback_to_backbone_on_saturation: true
```

### Parameters

- **log_every_n_steps** (int, default=100): How often to log the gram matrix
- **feature_key** (str, default="teacher_cls_token"): Which feature tensor to use from model outputs. Options include:
  - `"teacher_cls_token"`: Teacher network's CLS token features
  - `"student_cls_token"`: Student network's CLS token features
  - `"student_glob_cls_token"`: Student global view CLS tokens
- **max_samples** (int, default=128): Maximum number of samples to include in the gram matrix to prevent out-of-memory errors
- **saturation_offdiag_threshold** (float, default=0.995): Off-diagonal cosine threshold used to flag suspicious saturation.
- **saturation_offdiag_std_threshold** (float, default=0.01): Maximum off-diagonal standard deviation considered near-constant.
- **saturation_sample_var_threshold** (float, default=1e-6): Maximum across-sample variance to flag possible collapse.
- **auto_fallback_to_backbone_on_saturation** (bool, default=True): When projection-head features saturate, automatically retries with backbone feature keys to isolate framework-path vs model-side collapse.

### Implementation Details

1. **DDP Support**: The callback properly gathers features across all distributed ranks using `torch.distributed.all_gather`, so the gram matrix shows similarity across the entire batch (not just the local rank's portion).

2. **Memory Efficient**: 
   - Limits the number of samples to `max_samples` to prevent OOM
   - Cleans up intermediate tensors after use
   - Only logs on rank 0 to avoid redundant operations

3. **Error Handling**: 
   - Gracefully handles errors without interrupting training
   - Validates batch structure and feature dimensions
   - Safe nested dictionary access
   - Logs warnings for debugging

4. **Framework Root-Cause Diagnostics**:
   - Computes saturation statistics (`offdiag_mean`, `offdiag_std`, sample variance)
   - Emits debug metrics under `gram_matrix_debug/*`
   - Emits stage-level diagnostics for:
     - `gram_matrix_debug/stage_feature_extraction_*`
     - `gram_matrix_debug/stage_normalization_*`
     - `gram_matrix_debug/stage_sampling_*`
     - `gram_matrix_debug/stage_gather_*`
   - Differentiates likely framework-path saturation from likely model-side collapse
   - Can automatically fallback to backbone keys (for example `teacher_cls_token_backbone`) when projected features look framework-saturated
   - Skips image and telemetry logging gracefully when `WANDB_DISABLED` or `WANDB_MODE=disabled`

5. **Visualization**: Creates a heatmap with:
   - Viridis colormap (good for colorblind accessibility)
   - Fixed color scale from -1 to 1
   - Colorbar showing cosine similarity
   - Sample indices on both axes

### Example Output

The callback logs to wandb under the key `"gram_matrix"`. You'll see a heatmap where:
- The diagonal is always bright (each sample is identical to itself)
- Off-diagonal elements show cross-sample similarity
- Patterns in the matrix can reveal clustering or diversity in the learned features
- Additional framework diagnostics are logged under `gram_matrix_debug/*` when logger telemetry is available

### Troubleshooting

If the gram matrix is not appearing in wandb:

1. Ensure you're using `WandbLogger` in your trainer configuration
2. Check that the `feature_key` matches a key in your model's outputs
3. Look for warning messages in the training logs
4. Verify that matplotlib is installed (`matplotlib>=3.5.0,<4.0.0`)
5. If values are constantly near `1.0`, inspect `gram_matrix_debug/root_cause_*`:
   - `root_cause_framework=1` suggests projection-path saturation and fallback correction was applied
   - `root_cause_model=1` suggests true model-side collapse persisted even on backbone features
6. Use stage keys to localize saturation:
   - `stage_feature_extraction_*`: raw extracted feature signal
   - `stage_normalization_*`: post-normalization signal
   - `stage_sampling_*`: post-sampling signal
   - `stage_gather_*`: post-DDP merge signal used for final gram matrix

### Advanced Usage

To use different features or customize the logging:

```python
from callbacks import GramMatrixCallback

# Log student features instead of teacher features
callback = GramMatrixCallback(
    log_every_n_steps=50,
    feature_key="student_cls_token",
    max_samples=64  # Smaller matrix for faster logging
)
```
