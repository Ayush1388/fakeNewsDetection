from pathlib import Path

import torch
from tqdm import tqdm
from sklearn.metrics import f1_score


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        save_path,
    ):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_path = Path(save_path)

        self.save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.best_f1 = -1.0

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

        predictions = []
        labels = []

        iterator = tqdm(
            loader,
            leave=False,
            desc="Train" if training else "Valid",
        )

        for batch in iterator:

            input_ids = batch["input_ids"].to(
                self.device
            )

            attention_mask = batch[
                "attention_mask"
            ].to(self.device)

            linguistic_features = batch[
                "linguistic_features"
            ].to(self.device)

            target = batch["labels"].to(
                self.device
            )

            if training:
                self.optimizer.zero_grad(
                    set_to_none=True
                )

            with torch.set_grad_enabled(training):

                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    linguistic_features=linguistic_features,
                )

                loss = self.criterion(
                    output["logits"],
                    target,
                )

                if training:

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=1.0,
                    )

                    self.optimizer.step()

            total_loss += loss.item()

            pred = output["logits"].argmax(
                dim=-1
            )

            predictions.extend(
                pred.detach().cpu().numpy()
            )

            labels.extend(
                target.detach().cpu().numpy()
            )

        macro_f1 = f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )

        return {
            "loss": total_loss / max(len(loader), 1),
            "macro_f1": macro_f1,
        }

    def fit(
        self,
        train_loader,
        validation_loader,
        epochs,
    ):

        history = []

        for epoch in range(
            1,
            epochs + 1,
        ):

            train_metrics = self._run_epoch(
                train_loader,
                training=True,
            )

            val_metrics = self._run_epoch(
                validation_loader,
                training=False,
            )

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_f1": train_metrics["macro_f1"],
                    "val_loss": val_metrics["loss"],
                    "val_f1": val_metrics["macro_f1"],
                }
            )

            print(f"\nEpoch {epoch}/{epochs}")

            print(
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train F1: {train_metrics['macro_f1']:.4f}"
            )

            print(
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val F1: {val_metrics['macro_f1']:.4f}"
            )

            if val_metrics["macro_f1"] > self.best_f1:

                self.best_f1 = val_metrics["macro_f1"]

                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "val_f1": self.best_f1,
                    },
                    self.save_path,
                )

                print(
                    f"Saved best model → {self.save_path}"
                )

        return history