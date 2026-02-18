class DummyLogger:
    def __init__(self):
        self.records = []

    def log_metrics(self, metrics, step=0):
        self.records.append((metrics, step))


class DummyTrainer:
    LAST_INSTANCE = None

    def __init__(self, logger=None, **kwargs):
        del kwargs
        self.logger = logger or DummyLogger()
        DummyTrainer.LAST_INSTANCE = self

    def fit(self, lightning_module, data_module):
        del lightning_module, data_module
        self.logger.log_metrics({"train_total_loss": 0.0}, step=0)


class DummyLightningModule:
    def __init__(self, **kwargs):
        del kwargs


class DummyDataModule:
    def __init__(self, **kwargs):
        del kwargs
