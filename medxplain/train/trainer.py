"""Fine-tuning loop for the clinical risk classifier.

Written by hand rather than using ``transformers.Trainer`` so that the pieces
reviewers ask about are visible in one file: the early-stopping criterion, what
exactly gets restored as "the best model", how the class imbalance enters the
loss, and which quantity the learning-rate schedule follows.

Model selection uses **validation AUPRC by default, not accuracy or loss**. With
a 16% positive rate, validation loss is dominated by the majority class and
routinely selects a checkpoint that ranks the minority class worse; AUPRC is the
quantity the paper reports as its headline rare-event metric, so that is what
early stopping optimises.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from medxplain.data.dataset import ClinicalNoteDataset, Collator, class_weights
from medxplain.models.losses import build_loss
from medxplain.models.transformer import ClinicalRiskModel, ModelConfig, load_backbone
from medxplain.utils import get_logger, set_seed

LOGGER = get_logger("medxplain.train")


@dataclass
class TrainConfig:
    # optimisation
    learning_rate: float = 2e-5
    head_learning_rate: float | None = None  # defaults to 10x lr for the classifier head
    weight_decay: float = 0.01
    batch_size: int = 16
    eval_batch_size: int = 32
    epochs: int = 4
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    # imbalance handling
    loss: str = "weighted_ce"  # ce | weighted_ce | focal
    class_weight_scheme: str = "balanced"  # balanced | sqrt_balanced | none
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    # selection / stopping
    metric_for_best: str = "auprc"  # auprc | auroc | f1 | loss
    patience: int = 2
    eval_every_n_steps: int | None = None  # None = once per epoch
    # runtime
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    max_train_steps: int | None = None  # cap for smoke tests
    log_every: int = 25
    amp: bool = False
    extra: dict = field(default_factory=dict)

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainResult:
    best_metric: float
    best_epoch: int
    history: list[dict]
    train_seconds: float
    steps: int
    config: dict
    val_predictions: np.ndarray | None = None


def _binary_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """Threshold-free metrics plus F1 at 0.5, used for training-time monitoring."""
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    out: dict = {}
    # a batch of validation data can be single-class early in training
    if len(np.unique(y_true)) > 1:
        out["auroc"] = float(roc_auc_score(y_true, probs))
        out["auprc"] = float(average_precision_score(y_true, probs))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    out["f1"] = float(f1_score(y_true, (probs >= 0.5).astype(int), zero_division=0))
    return out


class Trainer:
    """Fine-tunes a :class:`ClinicalRiskModel` on one train/val split."""

    def __init__(
        self,
        model_cfg: ModelConfig,
        train_cfg: TrainConfig,
        tokenizer=None,
        model: ClinicalRiskModel | None = None,
    ):
        self.model_cfg = model_cfg
        self.cfg = train_cfg
        self.device = torch.device(train_cfg.resolved_device())

        if model is None:
            encoder, tok, hf_config = load_backbone(model_cfg)
            self.tokenizer = tokenizer or tok
            self.model = ClinicalRiskModel(model_cfg, encoder=encoder, hidden_size=hf_config.hidden_size)
        else:
            self.model = model
            self.tokenizer = tokenizer
        self.model.to(self.device)

    # -- data -------------------------------------------------------------------------

    def _loader(self, texts, labels, shuffle: bool, batch_size: int) -> DataLoader:
        dataset = ClinicalNoteDataset(
            texts,
            labels,
            self.tokenizer,
            max_length=self.model_cfg.max_length,
            truncation_strategy=self.model_cfg.truncation_strategy,
            head_fraction=self.model_cfg.head_fraction,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=Collator(self.tokenizer.pad_token_id),
            num_workers=self.cfg.num_workers,
            drop_last=False,
        )

    # -- optimisation ------------------------------------------------------------------

    def _optimizer(self):
        """AdamW with no weight decay on biases/LayerNorm, and a faster head LR.

        The randomly initialised classification head needs a larger step size
        than the pretrained encoder; using one LR for both either under-trains
        the head or destroys the pretrained representation.
        """
        head_lr = self.cfg.head_learning_rate or self.cfg.learning_rate * 10
        no_decay = ("bias", "LayerNorm.weight", "layer_norm")
        groups = [
            {
                "params": [
                    p for n, p in self.model.encoder.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.cfg.weight_decay,
                "lr": self.cfg.learning_rate,
            },
            {
                "params": [
                    p for n, p in self.model.encoder.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": self.cfg.learning_rate,
            },
            {
                "params": list(self.model.classifier.parameters()),
                "weight_decay": self.cfg.weight_decay,
                "lr": head_lr,
            },
        ]
        return torch.optim.AdamW(groups)

    @staticmethod
    def _scheduler(optimizer, total_steps: int, warmup_ratio: float):
        warmup = max(1, int(total_steps * warmup_ratio))

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / warmup
            progress = (step - warmup) / max(1, total_steps - warmup)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # -- evaluation --------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, texts: Sequence[str], labels: Sequence[int] | None = None) -> dict:
        loader = self._loader(texts, labels, shuffle=False, batch_size=self.cfg.eval_batch_size)
        self.model.eval()
        probs, ys, losses = [], [], []
        loss_fn = torch.nn.CrossEntropyLoss()
        for batch in loader:
            batch = batch.to(self.device)
            logits = self.model(**batch.as_model_inputs())["logits"]
            probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
            if batch.labels is not None:
                ys.append(batch.labels.cpu().numpy())
                losses.append(float(loss_fn(logits, batch.labels)))
        probs_arr = np.concatenate(probs)
        result = {"probabilities": probs_arr}
        if ys:
            y = np.concatenate(ys)
            result.update(_binary_metrics(y, probs_arr))
            result["loss"] = float(np.mean(losses))
            result["labels"] = y
        return result

    # -- training ----------------------------------------------------------------------

    def train(
        self,
        train_texts: Sequence[str],
        train_labels: Sequence[int],
        val_texts: Sequence[str],
        val_labels: Sequence[int],
        checkpoint_dir: str | Path | None = None,
        on_epoch_end: Callable[[int, dict], None] | None = None,
    ) -> TrainResult:
        set_seed(self.cfg.seed)
        started = time.perf_counter()

        train_loader = self._loader(train_texts, train_labels, True, self.cfg.batch_size)
        weights = class_weights(train_labels, self.cfg.class_weight_scheme).to(self.device)
        loss_fn = build_loss(
            self.cfg.loss,
            class_weights=weights,
            focal_gamma=self.cfg.focal_gamma,
            label_smoothing=self.cfg.label_smoothing,
        )
        LOGGER.info(
            "training %s | %d train / %d val | loss=%s weights=%s | device=%s",
            self.model_cfg.name, len(train_texts), len(val_texts), self.cfg.loss,
            [round(float(w), 3) for w in weights], self.device,
        )

        steps_per_epoch = math.ceil(len(train_loader) / self.cfg.gradient_accumulation_steps)
        total_steps = steps_per_epoch * self.cfg.epochs
        if self.cfg.max_train_steps:
            total_steps = min(total_steps, self.cfg.max_train_steps)
        optimizer = self._optimizer()
        scheduler = self._scheduler(optimizer, total_steps, self.cfg.warmup_ratio)
        scaler = torch.amp.GradScaler("cuda", enabled=self.cfg.amp and self.device.type == "cuda")

        higher_is_better = self.cfg.metric_for_best != "loss"
        best = -np.inf if higher_is_better else np.inf
        best_epoch, bad_epochs, global_step = -1, 0, 0
        history: list[dict] = []
        best_state: dict | None = None
        best_val_probs: np.ndarray | None = None
        stop = False

        for epoch in range(self.cfg.epochs):
            self.model.train()
            running, seen = 0.0, 0
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(train_loader):
                batch = batch.to(self.device)
                with torch.amp.autocast("cuda", enabled=self.cfg.amp and self.device.type == "cuda"):
                    logits = self.model(**batch.as_model_inputs())["logits"]
                    loss = loss_fn(logits, batch.labels) / self.cfg.gradient_accumulation_steps

                scaler.scale(loss).backward() if scaler.is_enabled() else loss.backward()
                running += loss.detach().item() * self.cfg.gradient_accumulation_steps
                seen += 1

                if (step + 1) % self.cfg.gradient_accumulation_steps == 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if global_step % self.cfg.log_every == 0:
                        LOGGER.info(
                            "  epoch %d step %d/%d | loss %.4f | lr %.2e",
                            epoch + 1, global_step, total_steps, running / max(seen, 1),
                            scheduler.get_last_lr()[0],
                        )
                    if self.cfg.max_train_steps and global_step >= self.cfg.max_train_steps:
                        stop = True
                        break

            metrics = self.evaluate(val_texts, val_labels)
            record = {
                "epoch": epoch + 1,
                "train_loss": running / max(seen, 1),
                "val_loss": metrics["loss"],
                "val_auroc": metrics["auroc"],
                "val_auprc": metrics["auprc"],
                "val_f1": metrics["f1"],
                "step": global_step,
            }
            history.append(record)
            LOGGER.info(
                "epoch %d | train loss %.4f | val loss %.4f | AUROC %.4f | AUPRC %.4f",
                epoch + 1, record["train_loss"], record["val_loss"],
                record["val_auroc"], record["val_auprc"],
            )
            if on_epoch_end:
                on_epoch_end(epoch + 1, record)

            current = metrics[self.cfg.metric_for_best] if self.cfg.metric_for_best != "loss" else metrics["loss"]
            improved = (current > best) if higher_is_better else (current < best)
            if improved and not (isinstance(current, float) and math.isnan(current)):
                best, best_epoch, bad_epochs = float(current), epoch + 1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                best_val_probs = metrics["probabilities"]
            else:
                bad_epochs += 1
                if bad_epochs >= self.cfg.patience:
                    LOGGER.info("early stopping at epoch %d (best epoch %d)", epoch + 1, best_epoch)
                    stop = True

            if stop:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        if checkpoint_dir is not None:
            self.model.save(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)

        elapsed = time.perf_counter() - started
        return TrainResult(
            best_metric=float(best),
            best_epoch=best_epoch,
            history=history,
            train_seconds=elapsed,
            steps=global_step,
            config={
                "model": self.model_cfg.__dict__ | {"hf_id": self.model_cfg.hf_id},
                "train": self.cfg.__dict__,
            },
            val_predictions=best_val_probs,
        )
