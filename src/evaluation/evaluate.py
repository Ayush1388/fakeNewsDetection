import json
from pathlib import Path

import numpy as np
import torch

from .metrics import (
    calculate_metrics,
    detailed_report,
    get_confusion_matrix,
)


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device,
    class_names,
    output_dir,
):

    model.eval()

    predictions = []
    labels = []
    confidences = []
    entropies = []

    for batch in loader:

        input_ids = batch["input_ids"].to(
            device
        )

        attention_mask = batch[
            "attention_mask"
        ].to(device)

        linguistic_features = batch[
            "linguistic_features"
        ].to(device)

        target = batch["labels"].to(
            device
        )

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            linguistic_features=linguistic_features,
        )

        pred = output["logits"].argmax(
            dim=-1
        )

        predictions.extend(
            pred.cpu().numpy()
        )

        labels.extend(
            target.cpu().numpy()
        )

        confidences.extend(
            output["confidence"]
            .cpu()
            .numpy()
        )

        entropies.extend(
            output["entropy"]
            .cpu()
            .numpy()
        )

    metrics = calculate_metrics(
        labels,
        predictions,
    )

    report = detailed_report(
        labels,
        predictions,
        class_names,
    )

    matrix = get_confusion_matrix(
        labels,
        predictions,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    with open(
        output_dir / "classification_report.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report)

    np.savetxt(
        output_dir / "confusion_matrix.csv",
        matrix,
        delimiter=",",
        fmt="%d",
    )

    np.savetxt(
        output_dir / "confidence.csv",
        np.array(confidences),
        delimiter=",",
    )

    np.savetxt(
        output_dir / "entropy.csv",
        np.array(entropies),
        delimiter=",",
    )

    print("\n===== FINAL TEST RESULTS =====")

    for key, value in metrics.items():
        print(
            f"{key}: {value:.4f}"
        )

    print("\n===== CLASSIFICATION REPORT =====")
    print(report)

    print(
        "\nConfusion matrix saved to:",
        output_dir / "confusion_matrix.csv",
    )

    return metrics