import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import matplotlib.pyplot as plt

from callbacks.gram_matrix_logger import GramMatrixCallback


class _FakeExperiment:
    def __init__(self):
        self.logged = []

    def log(self, payload, step=None):
        self.logged.append((payload, step))


class _FakeWandbLogger:
    def __init__(self):
        self.experiment = _FakeExperiment()


class _FakeTrainer:
    def __init__(self, global_step=1, world_size=1, global_rank=0, logger=None):
        self.global_step = global_step
        self.world_size = world_size
        self.global_rank = global_rank
        self.logger = logger


class _FakeLightningModule:
    def __init__(self, model_outputs):
        self._model_outputs = model_outputs
        self.device = torch.device("cpu")

    def model(self, _views):
        return self._model_outputs


class GramMatrixCallbackTests(unittest.TestCase):
    def test_logs_wandb_image_under_gram_matrix_key(self):
        callback = GramMatrixCallback()
        logger = _FakeWandbLogger()
        trainer = SimpleNamespace(global_step=10, logger=logger)
        gram_matrix = torch.eye(4)
        fake_wandb = types.SimpleNamespace(Image=lambda fig: ("wandb-image", fig))

        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            with patch("pytorch_lightning.loggers.WandbLogger", _FakeWandbLogger):
                with patch.dict(
                    os.environ, {"WANDB_DISABLED": "0", "WANDB_MODE": "online"}, clear=False
                ):
                    GramMatrixCallback._log_gram_matrix.__wrapped__(callback, trainer, gram_matrix)

        self.assertEqual(len(logger.experiment.logged), 1)
        payload, step = logger.experiment.logged[0]
        self.assertIn("gram_matrix", payload)
        self.assertEqual(step, 10)

    def test_cadence_skips_step_zero_and_non_interval_steps(self):
        callback = GramMatrixCallback(log_every_n_steps=10)
        pl_module = _FakeLightningModule({"pred": {"teacher_cls_token": torch.randn(8, 6)}})
        batch = (torch.randn(8, 6),)

        trainer_step_zero = _FakeTrainer(global_step=0, logger=_FakeWandbLogger())
        trainer_non_interval = _FakeTrainer(global_step=9, logger=_FakeWandbLogger())

        with patch.object(callback, "_log_gram_matrix") as log_mock:
            callback.on_train_batch_end(trainer_step_zero, pl_module, None, batch, 0)
            callback.on_train_batch_end(trainer_non_interval, pl_module, None, batch, 0)

        log_mock.assert_not_called()

    def test_feature_key_and_max_samples_are_honored(self):
        callback = GramMatrixCallback(
            log_every_n_steps=10,
            feature_key="student_cls_token",
            max_samples=4,
        )
        pl_module = _FakeLightningModule({"pred": {"student_cls_token": torch.randn(12, 8)}})
        trainer = _FakeTrainer(global_step=10, logger=_FakeWandbLogger())
        batch = (torch.randn(12, 8),)
        captured = {}

        def _capture_gram(_trainer, gram_matrix):
            captured["shape"] = tuple(gram_matrix.shape)

        with patch.object(callback, "_log_gram_matrix", side_effect=_capture_gram) as log_mock:
            callback.on_train_batch_end(trainer, pl_module, None, batch, 0)

        log_mock.assert_called_once()
        self.assertEqual(captured["shape"], (4, 4))

    def test_missing_feature_key_logs_warning_and_returns_safely(self):
        callback = GramMatrixCallback(log_every_n_steps=1, feature_key="missing_key")
        pl_module = _FakeLightningModule({"pred": {"teacher_cls_token": torch.randn(6, 5)}})
        trainer = _FakeTrainer(global_step=1, logger=_FakeWandbLogger())
        batch = (torch.randn(6, 5),)

        with self.assertLogs(level="WARNING") as logs:
            callback.on_train_batch_end(trainer, pl_module, None, batch, 0)

        self.assertTrue(
            any("Feature key 'missing_key'" in message for message in logs.output),
            f"Expected missing feature_key warning in logs, got: {logs.output}",
        )
        self.assertEqual(len(trainer.logger.experiment.logged), 1)
        self.assertIn("gram_matrix_error", trainer.logger.experiment.logged[0][0])

    def test_disabled_wandb_environment_short_circuits_logging(self):
        callback = GramMatrixCallback()
        logger = _FakeWandbLogger()
        trainer = SimpleNamespace(global_step=10, logger=logger)
        gram_matrix = torch.eye(3)
        fake_wandb = types.SimpleNamespace(Image=lambda fig: ("wandb-image", fig))

        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            with patch("pytorch_lightning.loggers.WandbLogger", _FakeWandbLogger):
                with patch.dict(os.environ, {"WANDB_DISABLED": "1"}, clear=False):
                    GramMatrixCallback._log_gram_matrix.__wrapped__(callback, trainer, gram_matrix)

        self.assertEqual(logger.experiment.logged, [])
        self.assertTrue(callback._warned_wandb_disabled)

    def test_wandb_mode_disabled_short_circuits_logging(self):
        callback = GramMatrixCallback()
        logger = _FakeWandbLogger()
        trainer = SimpleNamespace(global_step=10, logger=logger)
        gram_matrix = torch.eye(3)
        fake_wandb = types.SimpleNamespace(Image=lambda fig: ("wandb-image", fig))

        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            with patch("pytorch_lightning.loggers.WandbLogger", _FakeWandbLogger):
                with patch.dict(os.environ, {"WANDB_MODE": "disabled"}, clear=False):
                    GramMatrixCallback._log_gram_matrix.__wrapped__(callback, trainer, gram_matrix)

        self.assertEqual(logger.experiment.logged, [])
        self.assertTrue(callback._warned_wandb_disabled)

    def test_log_gram_matrix_closes_figure_when_wandb_disabled(self):
        callback = GramMatrixCallback()
        logger = _FakeWandbLogger()
        trainer = SimpleNamespace(global_step=10, logger=logger)
        gram_matrix = torch.eye(3)
        fake_wandb = types.SimpleNamespace(Image=lambda fig: ("wandb-image", fig))

        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            with patch("pytorch_lightning.loggers.WandbLogger", _FakeWandbLogger):
                with patch.dict(os.environ, {"WANDB_DISABLED": "1"}, clear=False):
                    with patch.object(plt, "close") as close_mock:
                        GramMatrixCallback._log_gram_matrix.__wrapped__(callback, trainer, gram_matrix)

        close_mock.assert_called_once()

    def test_ddp_all_reduce_gate_prevents_all_gather_on_invalid_rank(self):
        callback = GramMatrixCallback(log_every_n_steps=1)
        pl_module = _FakeLightningModule({"pred": {"teacher_cls_token": torch.randn(10, 8)}})
        trainer = _FakeTrainer(global_step=1, world_size=2, logger=_FakeWandbLogger())
        batch = (torch.randn(10, 8),)

        def _force_global_failure(tensor, op=None):
            del op
            tensor.fill_(0)

        with patch("callbacks.gram_matrix_logger.dist.is_available", return_value=True):
            with patch("callbacks.gram_matrix_logger.dist.is_initialized", return_value=True):
                with patch(
                    "callbacks.gram_matrix_logger.dist.all_reduce",
                    side_effect=_force_global_failure,
                ) as all_reduce_mock:
                    with patch("callbacks.gram_matrix_logger.dist.all_gather") as all_gather_mock:
                        with patch.object(callback, "_log_gram_matrix") as log_mock:
                            callback.on_train_batch_end(trainer, pl_module, None, batch, 0)

        all_reduce_mock.assert_called_once()
        all_gather_mock.assert_not_called()
        log_mock.assert_not_called()

    def test_ddp_gather_handles_uneven_rank_counts(self):
        callback = GramMatrixCallback(log_every_n_steps=1, max_samples=8)
        pl_module = _FakeLightningModule({"pred": {"student_cls_token": torch.randn(5, 4)}})
        trainer = _FakeTrainer(global_step=1, world_size=2, logger=_FakeWandbLogger())
        batch = (torch.randn(5, 4),)
        captured = {}

        def _all_gather_side_effect(output_list, input_tensor):
            # First all_gather call exchanges local counts.
            if input_tensor.numel() == 1:
                output_list[0].fill_(5)
                output_list[1].fill_(3)
                return
            # Second all_gather call exchanges padded feature tensors.
            output_list[0].copy_(input_tensor)
            output_list[1].zero_()
            output_list[1][:3, :].fill_(1.0)

        def _capture_gram(_trainer, gram_matrix):
            captured["shape"] = tuple(gram_matrix.shape)

        with patch("callbacks.gram_matrix_logger.dist.is_available", return_value=True):
            with patch("callbacks.gram_matrix_logger.dist.is_initialized", return_value=True):
                with patch(
                    "callbacks.gram_matrix_logger.dist.all_reduce",
                    side_effect=lambda tensor, op=None: None,
                ):
                    with patch(
                        "callbacks.gram_matrix_logger.dist.all_gather",
                        side_effect=_all_gather_side_effect,
                    ) as all_gather_mock:
                        with patch.object(callback, "_log_gram_matrix", side_effect=_capture_gram):
                            callback.on_train_batch_end(trainer, pl_module, None, batch, 0)

        self.assertEqual(all_gather_mock.call_count, 2)
        self.assertEqual(captured["shape"], (8, 8))

    def test_rejects_non_positive_log_every_n_steps(self):
        with self.assertRaises(ValueError):
            GramMatrixCallback(log_every_n_steps=0)

    def test_framework_saturation_uses_backbone_fallback(self):
        callback = GramMatrixCallback(
            log_every_n_steps=1,
            feature_key="teacher_cls_token",
            max_samples=8,
            saturation_offdiag_threshold=0.98,
            saturation_offdiag_std_threshold=0.05,
            saturation_sample_var_threshold=1e-4,
            auto_fallback_to_backbone_on_saturation=True,
        )
        primary = torch.ones(8, 4)  # framework-path saturation signature
        fallback = torch.randn(8, 4)
        pl_module = _FakeLightningModule(
            {
                "pred": {
                    "teacher_cls_token": primary,
                    "teacher_cls_token_backbone": fallback,
                }
            }
        )
        trainer = _FakeTrainer(global_step=1, logger=_FakeWandbLogger())
        batch = (torch.randn(8, 4),)

        with patch.object(callback, "_log_gram_matrix") as log_mock:
            callback.on_train_batch_end(trainer, pl_module, None, batch, 0)

        self.assertTrue(log_mock.called)
        debug_payloads = [p for p, _ in trainer.logger.experiment.logged if "gram_matrix_debug/fallback_used" in p]
        self.assertTrue(debug_payloads, "Expected debug payload with fallback_used metric")
        self.assertEqual(debug_payloads[-1]["gram_matrix_debug/fallback_used"], 1.0)
        self.assertEqual(debug_payloads[-1]["gram_matrix_debug/root_cause_framework"], 1.0)

    def test_model_side_collapse_reports_model_root_cause(self):
        callback = GramMatrixCallback(
            log_every_n_steps=1,
            feature_key="teacher_cls_token",
            max_samples=8,
            saturation_offdiag_threshold=0.98,
            saturation_offdiag_std_threshold=0.05,
            saturation_sample_var_threshold=1e-4,
            auto_fallback_to_backbone_on_saturation=True,
        )
        primary = torch.ones(8, 4)
        fallback = torch.ones(8, 4)
        pl_module = _FakeLightningModule(
            {
                "pred": {
                    "teacher_cls_token": primary,
                    "teacher_cls_token_backbone": fallback,
                }
            }
        )
        trainer = _FakeTrainer(global_step=1, logger=_FakeWandbLogger())
        batch = (torch.randn(8, 4),)

        with patch.object(callback, "_log_gram_matrix") as log_mock:
            callback.on_train_batch_end(trainer, pl_module, None, batch, 0)

        self.assertTrue(log_mock.called)
        debug_payloads = [p for p, _ in trainer.logger.experiment.logged if "gram_matrix_debug/root_cause_model" in p]
        self.assertTrue(debug_payloads, "Expected debug payload with root-cause metrics")
        self.assertEqual(debug_payloads[-1]["gram_matrix_debug/fallback_used"], 0.0)
        self.assertEqual(debug_payloads[-1]["gram_matrix_debug/root_cause_model"], 1.0)

    def test_stage_diagnostics_include_pipeline_metrics(self):
        callback = GramMatrixCallback(
            log_every_n_steps=1,
            feature_key="teacher_cls_token",
            max_samples=8,
            saturation_offdiag_threshold=0.98,
            saturation_offdiag_std_threshold=0.05,
            saturation_sample_var_threshold=1e-4,
            auto_fallback_to_backbone_on_saturation=True,
        )
        primary = torch.ones(8, 4)
        fallback = torch.randn(8, 4)
        pl_module = _FakeLightningModule(
            {
                "pred": {
                    "teacher_cls_token": primary,
                    "teacher_cls_token_backbone": fallback,
                }
            }
        )
        trainer = _FakeTrainer(global_step=1, logger=_FakeWandbLogger())
        batch = (torch.randn(8, 4),)

        with patch.object(callback, "_log_gram_matrix"):
            callback.on_train_batch_end(trainer, pl_module, None, batch, 0)

        debug_payloads = [
            p for p, _ in trainer.logger.experiment.logged if "gram_matrix_debug/fallback_used" in p
        ]
        self.assertTrue(debug_payloads, "Expected debug payload with saturation diagnostics")
        payload = debug_payloads[-1]

        expected_stage_keys = {
            "gram_matrix_debug/stage_feature_extraction_offdiag_mean",
            "gram_matrix_debug/stage_feature_extraction_offdiag_std",
            "gram_matrix_debug/stage_feature_extraction_sample_variance",
            "gram_matrix_debug/stage_normalization_offdiag_mean",
            "gram_matrix_debug/stage_normalization_offdiag_std",
            "gram_matrix_debug/stage_normalization_sample_variance",
            "gram_matrix_debug/stage_sampling_offdiag_mean",
            "gram_matrix_debug/stage_sampling_offdiag_std",
            "gram_matrix_debug/stage_sampling_sample_variance",
            "gram_matrix_debug/stage_gather_offdiag_mean",
            "gram_matrix_debug/stage_gather_offdiag_std",
            "gram_matrix_debug/stage_gather_sample_variance",
        }
        self.assertTrue(
            expected_stage_keys.issubset(payload.keys()),
            f"Missing expected stage diagnostics keys. Payload keys: {sorted(payload.keys())}",
        )

    def test_disabled_wandb_skips_diagnostics_telemetry(self):
        callback = GramMatrixCallback(log_every_n_steps=1, feature_key="missing_key")
        pl_module = _FakeLightningModule({"pred": {"teacher_cls_token": torch.randn(6, 5)}})
        trainer = _FakeTrainer(global_step=1, logger=_FakeWandbLogger())
        batch = (torch.randn(6, 5),)

        with patch.dict(os.environ, {"WANDB_DISABLED": "1"}, clear=False):
            callback.on_train_batch_end(trainer, pl_module, None, batch, 0)

        self.assertEqual(
            trainer.logger.experiment.logged,
            [],
            "Expected no telemetry payloads when WANDB is disabled",
        )

    def test_adaptive_colormap_title_contains_offdiag_statistics(self):
        """Regression: fixed vmin=-1 made cosine range [0.95,1.0] look identical to 1.0."""
        callback = GramMatrixCallback()
        logger = _FakeWandbLogger()
        trainer = types.SimpleNamespace(global_step=42, logger=logger)
        # Gram matrix where off-diagonal is 0.97 — visually distinct from 1.0 but
        # previously hidden by the fixed vmin=-1 colormap.
        n = 4
        gram_matrix = torch.full((n, n), 0.97)
        gram_matrix.fill_diagonal_(1.0)
        fake_wandb = types.SimpleNamespace(Image=lambda fig: ("wandb-image", fig))
        captured_titles = []

        import matplotlib.axes as mpl_axes

        original_set_title = mpl_axes.Axes.set_title

        def _capture_title(self, title, **kwargs):
            captured_titles.append(title)
            return original_set_title(self, title, **kwargs)

        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            with patch("pytorch_lightning.loggers.WandbLogger", _FakeWandbLogger):
                with patch.dict(
                    os.environ, {"WANDB_DISABLED": "0", "WANDB_MODE": "online"}, clear=False
                ):
                    with patch.object(mpl_axes.Axes, "set_title", _capture_title):
                        GramMatrixCallback._log_gram_matrix.__wrapped__(
                            callback, trainer, gram_matrix
                        )

        self.assertTrue(captured_titles, "Expected ax.set_title to be called")
        title = captured_titles[0]
        # Title must include off-diagonal statistics so researcher sees actual numbers,
        # not just a flat yellow image.
        self.assertIn("off-diag mean", title, f"Title missing 'off-diag mean': {title!r}")
        self.assertIn("std", title, f"Title missing 'std': {title!r}")
        self.assertIn("cmap", title, f"Title missing 'cmap' range annotation: {title!r}")

    def test_backbone_saturation_detected_with_corrected_threshold(self):
        """Regression: 1e-6 threshold missed backbone (768-dim) saturation at cosine ~0.999.
        The corrected default of 1e-4 catches it so model-side collapse is reported."""
        # 768-dim backbone-like features with cosine ~0.9999 (all nearly identical vectors).
        dim = 768
        base = torch.randn(1, dim)
        base = base / base.norm(dim=-1, keepdim=True)
        tiny_noise = torch.randn(8, dim) * 1e-4
        primary = base.expand(8, -1) + tiny_noise  # cosine ~= 0.9999

        callback = GramMatrixCallback(
            log_every_n_steps=1,
            feature_key="teacher_cls_token",
            max_samples=8,
            # Use default saturation_sample_var_threshold (now 1e-4) to test the default
        )
        pl_module = _FakeLightningModule(
            {
                "pred": {
                    "teacher_cls_token": primary,
                    "teacher_cls_token_backbone": primary.clone(),
                }
            }
        )
        trainer = _FakeTrainer(global_step=1, logger=_FakeWandbLogger())
        batch = (torch.randn(8, dim),)

        with patch.object(callback, "_log_gram_matrix"):
            callback.on_train_batch_end(trainer, pl_module, None, batch, 0)

        debug_payloads = [
            p for p, _ in trainer.logger.experiment.logged
            if "gram_matrix_debug/root_cause_model" in p
        ]
        self.assertTrue(
            debug_payloads,
            "Expected model-side saturation to be detected for 768-dim near-identical features",
        )
        self.assertEqual(debug_payloads[-1]["gram_matrix_debug/root_cause_model"], 1.0)
