import os

import torch
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm


# Keys that describe the target / bookkeeping rather than a model
# input, so they must never be forwarded into model(**kwargs).
NON_MODEL_KEYS = {"labels", "ids", "texts", "id", "text", "label"}


class Trainer:
    """
    Generic trainer.

    Earlier versions of this trainer hardcoded the propagation-model
    batch layout (``node_features`` / ``edge_index`` / ``delays``),
    which meant it silently crashed (KeyError) on any batch produced
    by the plain ``FakeNewsDataset`` used by the DeBERTa baseline and
    TEG-FND models. The trainer now builds its model kwargs directly
    from whatever keys a batch actually contains, so it works
    unchanged for all three model variants (deberta / tegfnd /
    propagation).
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        save_path,
        scheduler=None,
        max_grad_norm=1.0,
        grad_accum_steps=1,
        patience=5,
        aux_loss_key="aux_loss",
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_path = save_path
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.patience = patience
        self.aux_loss_key = aux_loss_key

        self.use_amp = self.device.type == "cuda"

        if self.use_amp:
            self.scaler = torch.amp.GradScaler(
                "cuda",
                enabled=True,
            )
        else:
            self.scaler = None

    def _move_batch_to_device(self, batch):
        model_kwargs = {}
        labels = None

        for key, value in batch.items():

            if key == "labels":
                labels = value.to(self.device, non_blocking=True)
                continue

            if key in NON_MODEL_KEYS:
                continue

            if torch.is_tensor(value):
                model_kwargs[key] = value.to(
                    self.device,
                    non_blocking=True,
                )
            elif isinstance(value, (list, tuple)) and value and torch.is_tensor(value[0]):
                model_kwargs[key] = [
                    item.to(self.device, non_blocking=True)
                    for item in value
                ]
            else:
                model_kwargs[key] = value

        return model_kwargs, labels

    def _forward_loss(self, batch):
        model_kwargs, labels = self._move_batch_to_device(batch)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self.use_amp,
        ):
            output = self.model(**model_kwargs)

            loss = self.criterion(
                output["logits"],
                labels,
            )

            aux_loss = output.get(self.aux_loss_key)

            if aux_loss is not None:
                loss = loss + aux_loss

        return loss, output, labels

    def _run_epoch(
        self,
        loader,
        training=True,
    ):
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        all_predictions = []
        all_labels = []

        progress = tqdm(
            loader,
            desc="Train" if training else "Val",
        )

        if training:
            self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(progress):

            if training:
                loss, output, labels = self._forward_loss(batch)

                scaled_loss = loss / self.grad_accum_steps

                if self.use_amp:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                is_accum_boundary = (
                    (step + 1) % self.grad_accum_steps == 0
                    or (step + 1) == len(loader)
                )

                if is_accum_boundary:

                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm,
                    )

                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad(set_to_none=True)

                    if self.scheduler is not None:
                        self.scheduler.step()

            else:
                with torch.no_grad():
                    loss, output, labels = self._forward_loss(batch)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite loss detected: "
                    f"{loss.item()}"
                )

            predictions = torch.argmax(
                output["logits"],
                dim=1,
            )

            total_loss += loss.item()

            all_predictions.extend(
                predictions.detach()
                .cpu()
                .tolist()
            )

            all_labels.extend(
                labels.detach()
                .cpu()
                .tolist()
            )

            current_f1 = f1_score(
                all_labels,
                all_predictions,
                average="macro",
                zero_division=0,
            )

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                f1=f"{current_f1:.4f}",
            )

        average_loss = (
            total_loss / len(loader)
        )

        epoch_f1 = f1_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        )

        epoch_accuracy = accuracy_score(
            all_labels,
            all_predictions,
        )

        return average_loss, epoch_f1, epoch_accuracy

    def fit(
        self,
        train_loader,
        validation_loader,
        epochs,
    ):
        best_val_f1 = -float("inf")
        epochs_without_improvement = 0

        history = {
            "train_loss": [],
            "train_f1": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_f1": [],
            "val_accuracy": [],
        }

        os.makedirs(
            os.path.dirname(self.save_path) or ".",
            exist_ok=True,
        )

        for epoch in range(
            1,
            epochs + 1,
        ):
            print(
                f"\nEpoch {epoch}/{epochs}"
            )

            train_loss, train_f1, train_accuracy = (
                self._run_epoch(
                    train_loader,
                    training=True,
                )
            )

            val_loss, val_f1, val_accuracy = (
                self._run_epoch(
                    validation_loader,
                    training=False,
                )
            )

            history["train_loss"].append(train_loss)
            history["train_f1"].append(train_f1)
            history["train_accuracy"].append(train_accuracy)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)
            history["val_accuracy"].append(val_accuracy)

            print(
                f"Train Loss: {train_loss:.4f} "
                f"| Train F1: {train_f1:.4f} "
                f"| Train Acc: {train_accuracy:.4f}"
            )

            print(
                f"Val Loss: {val_loss:.4f} "
                f"| Val F1: {val_f1:.4f} "
                f"| Val Acc: {val_accuracy:.4f}"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                epochs_without_improvement = 0

                torch.save(
                    {
                        "model_state_dict":
                            self.model.state_dict(),
                        "optimizer_state_dict":
                            self.optimizer.state_dict(),
                        "epoch": epoch,
                        "val_f1": val_f1,
                        "val_accuracy": val_accuracy,
                    },
                    self.save_path,
                )

                print(
                    "Saved best checkpoint: "
                    f"{self.save_path}"
                )
            else:
                epochs_without_improvement += 1

                print(
                    "No val F1 improvement for "
                    f"{epochs_without_improvement} epoch(s)."
                )

                if (
                    self.patience is not None
                    and epochs_without_improvement >= self.patience
                ):
                    print(
                        "Early stopping triggered "
                        f"(patience={self.patience})."
                    )
                    break

        history["best_val_f1"] = best_val_f1

        return history
