"""
General module for training machine learning models.

This module provides a unified interface for training PyTorch deep learning models
with built-in support for metrics tracking, tensorboard logging, early stopping,
and model checkpointing.

Before training, DLModule has to be configured using the .configure() method.
Configuration requires: loss function, optimizer, scheduler (optional),
metrics (optional), arguments, and device (optional).

For training, the .fit() method has to be called with train and validation DataLoaders
and the number of epochs.

For evaluation, the .evaluate() method can be used with a DataLoader to assess
model performance on a dataset.

For inference, the .predict() method generates predictions on new data using
a DataLoader.
"""

import os
import re
import time
import torch
import argparse
import torchmetrics
import torch.utils.tensorboard
import datetime
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
        args: argparse.Namespace = {},
        device: str | None = "auto"
    ) -> Self:

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.device = self._get_auto_device() if device == "auto" else torch.device(device)
        self.metrics = metrics if metrics is not None else {}
        self._create_log_name(args)
        self.epoch = 0
        self._tb_writers = {}
        self.to(self.device)

        # move merics to device
        for value in metrics.values():
            value.to(self.device)

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

            train_loss = self.average_loss
            self._write_tb_log()
            self.evaluate(dev_loader)
            self._write_tb_log()
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

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        print_loss: bool = False,
        epochs: int = 0,
    ) -> None:

        self.eval()
        self._reset_loss()
        self._reset_metrics()

        for x, y in dataloader:
            self.eval_step(x, y)

        dev_metrics_log = []
        for name, metric in self.metrics.items():
            dev_metrics_log.append(f"dev_{name}: {metric.compute().item():.4f}")
        if self.metrics: print(f"{', '.join(dev_metrics_log)}")

        if print_loss: print(f"Evaluation - dev_loss: {self.average_loss:.4}")
        
        while self.epoch < epochs:
            self.epoch += 1
            self._write_tb_log()

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

    def predict(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> list [torch.Tensor]:

        self.eval()
        y_preds = []

        for x, _ in dataloader:
            x = x.to(self.device)
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
        path_str: str
    ) -> None:
        state_dict = self.state_dict()
        full_path = os.path.join(path_str, self.model_name, "best_weights.pt")
        os.path.dirname(full_path) and os.makedirs(os.path.dirname(full_path), exist_ok=True)
        torch.save(state_dict, full_path)


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
            self._tb_writers[name] = torch.utils.tensorboard.SummaryWriter(os.path.join(self.logdir, self.model_name, name, self.args_name))
        return self._tb_writers[name]


    def _eval_stop_and_weights(self) -> None:
        curr_metric = self.average_loss if self.eval_metric == 'loss' else self.metrics[self.eval_metric].compute()
        improved = curr_metric > self.best_metric_val if self.eval_metric_asc else curr_metric < self.best_metric_val

        if improved:
            self.best_metric_val = curr_metric
            self.stagnation = 0
            if self._save_weights: self.save_weights(self.logdir)
        else:
            self.stagnation += 1
            self.stop = self.stagnation >= self.early_stop if self.early_stop else False


    def _create_log_name(
        self,
        args: argparse.Namespace
    ) -> None:
        self.logdir = args.logdir
        self.model_name = self.__class__.__name__
        self.args_name = "{}-{}".format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
        )


    @staticmethod
    def _get_auto_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")