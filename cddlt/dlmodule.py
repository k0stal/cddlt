import os
import time
import torch
import torchmetrics
import torch.utils.tensorboard
import tqdm

from typing import Self

class DLModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = None
        self.register_buffer("loss_total", torch.tensor(0.0, dtype=torch.float32), persistent=False)
        self.register_buffer("loss_count", torch.tensor(0, dtype=torch.int64), persistent=False)

    def configure(
        self,
        loss: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        metrics: dict[str, torchmetrics.Metric] | None = {},
        logdir: str | None = "logs",
        device: str | None = "auto"
    ) -> Self:

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.metrics = metrics if metrics is not None else {}
        self.device = self._get_auto_device() if device == "auto" else torch.device(device)
        self.epoch = 0
        self.logdir = logdir
        self._tb_writers = {}
        self.to(self.device)

        return self

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        dev_loader: torch.utils.data.DataLoader,
        epochs: int,
        save_weights: bool = True,
        early_stop: int = 0,
        eval_metric: str = 'loss',
        eval_metric_asc: bool = False
    ) -> None:

        assert eval_metric == 'loss' or eval_metric in self.metrics.keys(), f"Invalid evaluation metric: {eval_metric}."
        assert self.optimizer != None and \
            self.loss != None and \
            self.device != None, f"Model has to be configured before calling fit()."

        self.eval_metric = eval_metric
        self.eval_metric_asc = eval_metric_asc

        self.stop = False
        self.best_metric_val = 1e9 if eval_metric_asc == False else 0
        self.early_stop = early_stop
        self._save_weights = save_weights
        self.stagnation = 0

        while self.epoch < epochs and not self.stop:
            self.epoch += 1
            start = time.time()

            self.train()
            self._reset_loss()
            self._reset_metrics()

            train_pbar = tqdm.tqdm(train_loader, desc=f"{self.epoch}/{epochs}", leave=False)
            for x, y in train_pbar:
                self.train_step(x, y)

            if self.metrics:
                for name, metric in self.metrics.items():
                    print(f"train_{name}: {metric.compute():.4f}")

            train_loss = self.average_loss
            self.evaluate(dev_loader)
            self._eval_stop_and_weights()

            print(f"Epoch {self.epoch}/{epochs} - train_loss: {train_loss:.4f}, dev_loss: {self.average_loss:.4f} ({time.time() - start:.2f}s)")

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> None:

        x, y = x.to(self.device), y.to(self.device)
        y_pred = self(x)
        self.optimizer.zero_grad()
        loss = self.loss(y_pred, y)
        loss.backward()

        with torch.no_grad():
            self.optimizer.step()
            self.scheduler is not None and self.scheduler.step()
            self._update_loss(loss)
            self._update_metrics(y_pred, y)
            self._write_tb_log()

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        log_loss: bool = False
    ) -> None:

        self.eval()
        self._reset_loss()
        self._reset_metrics()

        for x, y in dataloader:
            self.eval_step(x, y)

        for name, metric in self.metrics.items():
            print(f"dev_{name}: {metric.compute().item():.4f}")

        if log_loss: print(f"Evaluation - dev_loss: {self.average_loss:.4}")

    def eval_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> None:

        with torch.no_grad():
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self(x)
            loss = self.loss(y_pred, y)
            self._update_loss(loss)
            self._update_metrics(y_pred, y)
            self._write_tb_log()

    def predict(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> list [torch.Tensor]:

        self.eval()
        y_preds = []

        for x, _ in dataloader:
            y_pred = self.predict_step(x)
            y_preds.append(y_pred)
        return y_preds


    def predict_step(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        with torch.no_grad():
            y_pred = self(x)
            return y_pred


    def save_weights(
        self,
        path: str
    ) -> None:
        state_dict = self.state_dict()
        os.path.dirname(path) and os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)


    def load_weights(
        self,
        model_path: str,
        device: torch.device | None = "auto"
    ) -> Self:
        if not self.device:
            self.device = self._get_auto_device() if device == "auto" else torch.device(device)
        self.load_state_dict(torch.load(os.path.join(model_path, "best_weights.pt"), map_location=self.device))


    def _reset_loss(self) -> None:
        self.loss_total.zero_()
        self.loss_count.zero_()


    def _update_loss(
        self,
        loss: float
    ) -> None:

        self.loss_total.add_(loss)
        self.loss_count.add_(1)
        self.average_loss = self.loss_total / self.loss_count


    def _reset_metrics(self) -> None:
        for metric in self.metrics.values():
            metric.reset()


    def _update_metrics(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor
    ) -> None:

        for metric in self.metrics.values():
            metric.update(y_pred, y)


    def _write_tb_log(self) -> None:
        assert self.logdir is not None

        process = 'train' if self.training else 'dev'
        writer = self._get_tb_writer(process)

        for metric_name, metric in self.metrics.items():
            writer.add_scalar(metric_name, metric.compute(), self.epoch)
        writer.add_scalar('loss', self.average_loss, self.epoch)
        writer.flush()

    def _get_tb_writer(
        self,
        name: str
    ) -> torch.utils.tensorboard.SummaryWriter:

        assert self.logdir is not None
        if name not in self._tb_writers:
            self._tb_writers[name] = torch.utils.tensorboard.SummaryWriter(os.path.join(self.logdir, name))
        return self._tb_writers[name]

    def _eval_stop_and_weights(self) -> None:
        curr_metric = self.average_loss if self.eval_metric == 'loss' else self.metrics[self.eval_metric].compute()
        improved = curr_metric > self.best_metric_val if self.eval_metric_asc else curr_metric < self.best_metric_val

        if improved:
            self.best_metric_val = curr_metric
            self.stagnation = 0
            if self._save_weights: self.save_weights(os.path.join(self.logdir, "best_weights.pt"))
        else:
            self.stagnation += 1
            self.stop = self.stagnation >= self.early_stop if self.early_stop else False

    @staticmethod
    def _get_auto_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")