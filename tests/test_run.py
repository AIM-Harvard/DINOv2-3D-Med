import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import run as run_module


class _FakeTrainer:
    def __init__(self):
        self.calls = []
        self.logger = _FakeLogger()

    def fit(self, lightning_module, data_module):
        self.calls.append((lightning_module, data_module))
        if hasattr(lightning_module, "emit_smoke_metric"):
            lightning_module.emit_smoke_metric(self.logger)


class _FakeLogger:
    def __init__(self):
        self.records = []

    def log_metrics(self, metrics, step):
        self.records.append((metrics, step))


class _SmokeLightningModule:
    def emit_smoke_metric(self, logger):
        logger.log_metrics({"train_total_loss": 0.1}, step=0)


class RunEntrypointTests(unittest.TestCase):
    def test_real_config_parser_smoke_logs_metric(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(__file__).resolve().parents[1]
            cfg1 = Path(tmpdir) / "train.yaml"
            cfg2 = Path(tmpdir) / "model.yaml"

            cfg1.write_text(
                "\n".join(
                    [
                        f"project: {project_root}",
                        "trainer:",
                        "  _target_: project.tests.fixtures.DummyTrainer",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            cfg2.write_text(
                "\n".join(
                    [
                        "lightning_module:",
                        "  _target_: project.tests.fixtures.DummyLightningModule",
                        "data_module:",
                        "  _target_: project.tests.fixtures.DummyDataModule",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            run_module.run("fit", f"{cfg1},{cfg2}")

            fixtures_module = importlib.import_module("project.tests.fixtures")
            trainer = fixtures_module.DummyTrainer.LAST_INSTANCE
            self.assertIsNotNone(trainer)
            self.assertEqual(len(trainer.logger.records), 1)
            metrics, step = trainer.logger.records[0]
            self.assertEqual(step, 0)
            self.assertIn("train_total_loss", metrics)

    def test_fit_wiring_uses_merged_config_paths_and_calls_trainer_fit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg1 = Path(tmpdir) / "train.yaml"
            cfg2 = Path(tmpdir) / "model.yaml"
            cfg1.write_text("trainer: {}\n", encoding="utf-8")
            cfg2.write_text("lightning_module: {}\n", encoding="utf-8")

            trainer = _FakeTrainer()
            lightning_module = object()
            data_module = object()

            parser = Mock()
            parser.get.side_effect = lambda key: {
                "project": tmpdir,
                "trainer": {"_target_": "x"},
                "lightning_module": {"_target_": "x"},
                "data_module": {"_target_": "x"},
            }[key]
            parser.get_parsed_content.side_effect = lambda key: {
                "trainer": trainer,
                "lightning_module": lightning_module,
                "data_module": data_module,
            }[key]

            with patch("scripts.run.ConfigParser", return_value=parser), patch(
                "scripts.run.import_module_from_path"
            ):
                run_module.run("fit", f"{cfg1},{cfg2}")

            parser.read_config.assert_called_once_with([str(cfg1), str(cfg2)])
            parser.parse.assert_called_once()
            parser.update.assert_called_once_with({})
            self.assertEqual(trainer.calls, [(lightning_module, data_module)])

    def test_smoke_fit_path_emits_first_step_metric_to_logger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Path(tmpdir) / "train.yaml"
            cfg.write_text("trainer: {}\n", encoding="utf-8")

            trainer = _FakeTrainer()
            lightning_module = _SmokeLightningModule()
            data_module = object()

            parser = Mock()
            parser.get.side_effect = lambda key: {
                "project": tmpdir,
                "trainer": {"_target_": "x"},
                "lightning_module": {"_target_": "x"},
                "data_module": {"_target_": "x"},
            }[key]
            parser.get_parsed_content.side_effect = lambda key: {
                "trainer": trainer,
                "lightning_module": lightning_module,
                "data_module": data_module,
            }[key]

            with patch("scripts.run.ConfigParser", return_value=parser), patch(
                "scripts.run.import_module_from_path"
            ):
                run_module.run("fit", str(cfg))

            self.assertEqual(len(trainer.logger.records), 1)
            metrics, step = trainer.logger.records[0]
            self.assertEqual(step, 0)
            self.assertIn("train_total_loss", metrics)

    def test_invalid_config_file_list_format_fails_fast(self):
        with self.assertRaises(ValueError) as ctx:
            run_module.run("fit", "a.yaml,,b.yaml")
        self.assertIn("comma-separated", str(ctx.exception))

    def test_missing_config_file_raises_actionable_error(self):
        missing = "/tmp/path/that/does/not/exist.yaml"
        with self.assertRaises(FileNotFoundError) as ctx:
            run_module.run("fit", missing)
        self.assertIn("Config file not found", str(ctx.exception))
        self.assertIn(missing, str(ctx.exception))

    def test_missing_required_sections_raise_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Path(tmpdir) / "train.yaml"
            cfg.write_text("trainer: {}\n", encoding="utf-8")

            parser = Mock()

            def _get(key):
                if key == "project":
                    return tmpdir
                if key == "trainer":
                    return {"_target_": "x"}
                if key == "data_module":
                    return {"_target_": "x"}
                raise KeyError(key)

            parser.get.side_effect = _get

            with patch("scripts.run.ConfigParser", return_value=parser), patch(
                "scripts.run.import_module_from_path"
            ):
                with self.assertRaises(ValueError) as ctx:
                    run_module.run("fit", str(cfg))

            message = str(ctx.exception)
            self.assertIn("Missing required config sections", message)
            self.assertIn("lightning_module", message)

    def test_missing_project_path_raises_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Path(tmpdir) / "train.yaml"
            cfg.write_text("trainer: {}\n", encoding="utf-8")

            parser = Mock()

            def _get(key):
                if key == "trainer":
                    return {"_target_": "x"}
                if key == "lightning_module":
                    return {"_target_": "x"}
                if key == "data_module":
                    return {"_target_": "x"}
                raise KeyError(key)

            parser.get.side_effect = _get

            with patch("scripts.run.ConfigParser", return_value=parser), patch(
                "scripts.run.import_module_from_path"
            ):
                with self.assertRaises(ValueError) as ctx:
                    run_module.run("fit", str(cfg))

            self.assertIn("Missing required 'project' path", str(ctx.exception))

    def test_project_import_failure_is_actionable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Path(tmpdir) / "train.yaml"
            cfg.write_text("trainer: {}\n", encoding="utf-8")

            parser = Mock()
            parser.get.side_effect = lambda key: {
                "project": tmpdir,
                "trainer": {"_target_": "x"},
                "lightning_module": {"_target_": "x"},
                "data_module": {"_target_": "x"},
            }[key]

            with patch("scripts.run.ConfigParser", return_value=parser), patch(
                "scripts.run.import_module_from_path",
                side_effect=FileNotFoundError("missing __init__.py"),
            ):
                with self.assertRaises(ValueError) as ctx:
                    run_module.run("fit", str(cfg))

            self.assertIn("Failed to load project module", str(ctx.exception))

    def test_data_module_init_failure_has_remediation_hint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Path(tmpdir) / "train.yaml"
            cfg.write_text("trainer: {}\n", encoding="utf-8")

            parser = Mock()
            parser.get.side_effect = lambda key: {
                "project": tmpdir,
                "trainer": {"_target_": "x"},
                "lightning_module": {"_target_": "x"},
                "data_module": {"_target_": "x"},
            }[key]

            trainer = _FakeTrainer()
            parser.get_parsed_content.side_effect = lambda key: {
                "trainer": trainer,
                "lightning_module": object(),
            }.get(key) if key != "data_module" else (_ for _ in ()).throw(
                RuntimeError("dataset.json not found")
            )

            with patch("scripts.run.ConfigParser", return_value=parser), patch(
                "scripts.run.import_module_from_path"
            ):
                with self.assertRaises(ValueError) as ctx:
                    run_module.run("fit", str(cfg))

            message = str(ctx.exception)
            self.assertIn("Failed to initialize 'data_module'", message)
            self.assertIn("dataset path", message)


if __name__ == "__main__":
    unittest.main()
