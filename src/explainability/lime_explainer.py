import torch
import numpy as np

from lime.lime_text import LimeTextExplainer


class FakeNewsExplainer:

    def __init__(
        self,
        model,
        tokenizer,
        device,
        max_length=192,
        class_names=None,
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

        self.class_names = (
            class_names
            or ["class_0", "class_1"]
        )

        self.explainer = LimeTextExplainer(
            class_names=self.class_names
        )

    def _predict(self, texts):

        self.model.eval()

        outputs = []

        for text in texts:

            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = encoded[
                "input_ids"
            ].to(self.device)

            attention_mask = encoded[
                "attention_mask"
            ].to(self.device)

            from src.data.features import (
                extract_linguistic_features
            )

            features = torch.tensor(
                extract_linguistic_features(text),
                dtype=torch.float32,
            ).unsqueeze(0).to(
                self.device
            )

            with torch.no_grad():

                result = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    linguistic_features=features,
                )

            outputs.append(
                result["probabilities"]
                .squeeze(0)
                .cpu()
                .numpy()
            )

        return np.asarray(outputs)

    def explain(
        self,
        text,
        num_features=10,
    ):

        explanation = (
            self.explainer.explain_instance(
                text,
                self._predict,
                num_features=num_features,
            )
        )

        return explanation.as_list()