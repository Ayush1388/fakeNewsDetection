import os

import torch
from sklearn.metrics import f1_score
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        save_path,
        scheduler=None,
        max_grad_norm=1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_path = save_path
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm

        self.use_amp = self.device.type == "cuda"

        if self.use_amp:
            self.scaler = torch.amp.GradScaler(
                "cuda",
                enabled=True,
            )
        else:
            self.scaler = None

    def _move_batch_to_device(self, batch):
        input_ids = batch["input_ids"].to(
            self.device,
            non_blocking=True,
        )

        attention_mask = batch["attention_mask"].to(
            self.device,
            non_blocking=True,
        )

        linguistic_features = batch[
            "linguistic_features"
        ].to(
            self.device,
            non_blocking=True,
        )

        labels = batch["labels"].to(
            self.device,
            non_blocking=True,
        )

        node_features = [
            nodes.to(
                self.device,
                non_blocking=True,
            )
            for nodes in batch["node_features"]
        ]

        edge_index = [
            edges.to(
                self.device,
                non_blocking=True,
            )
            for edges in batch["edge_index"]
        ]

        delays = [
            delay.to(
                self.device,
                non_blocking=True,
            )
            for delay in batch["delays"]
        ]

        return (
            input_ids,
            attention_mask,
            linguistic_features,
            node_features,
            edge_index,
            delays,
            labels,
        )

    def _forward_loss(self, batch):
        (
            input_ids,
            attention_mask,
            linguistic_features,
            node_features,
            edge_index,
            delays,
            labels,
        ) = self._move_batch_to_device(batch)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self.use_amp,
        ):
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                linguistic_features=linguistic_features,
                node_features=node_features,
                edge_index=edge_index,
                delays=delays,
            )

            loss = self.criterion(
                output["logits"],
                labels,
            )

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

        for batch in progress:

            if training:
                self.optimizer.zero_grad(
                    set_to_none=True
                )

                loss, output, labels = (
                    self._forward_loss(batch)
                )

                if self.use_amp:
                    self.scaler.scale(
                        loss
                    ).backward()

                    self.scaler.unscale_(
                        self.optimizer
                    )

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm,
                    )

                    self.scaler.step(
                        self.optimizer
                    )

                    self.scaler.update()

                else:
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm,
                    )

                    self.optimizer.step()

            else:
                with torch.no_grad():
                    loss, output, labels = (
                        self._forward_loss(batch)
                    )

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

        return average_loss, epoch_f1

    def fit(
        self,
        train_loader,
        val_loader,
        epochs,
    ):
        best_val_f1 = -float("inf")

        os.makedirs(
            os.path.dirname(self.save_path),
            exist_ok=True,
        )

        for epoch in range(
            1,
            epochs + 1,
        ):
            print(
                f"\nEpoch {epoch}/{epochs}"
            )

            train_loss, train_f1 = (
                self._run_epoch(
                    train_loader,
                    training=True,
                )
            )

            val_loss, val_f1 = (
                self._run_epoch(
                    val_loader,
                    training=False,
                )
            )

            if self.scheduler is not None:
                self.scheduler.step()

            print(
                f"Train Loss: {train_loss:.4f} "
                f"| Train F1: {train_f1:.4f}"
            )

            print(
                f"Val Loss: {val_loss:.4f} "
                f"| Val F1: {val_f1:.4f}"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1

                torch.save(
                    {
                        "model_state_dict":
                            self.model.state_dict(),
                        "optimizer_state_dict":
                            self.optimizer.state_dict(),
                        "epoch": epoch,
                        "val_f1": val_f1,
                    },
                    self.save_path,
                )

                print(
                    "Saved best checkpoint: "
                    f"{self.save_path}"
                )