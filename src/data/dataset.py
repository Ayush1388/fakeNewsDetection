import numpy as np
import torch

from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .features import extract_linguistic_features


class FakeNewsDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length=192,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        text = str(row["text"])
        label = int(row["label"])

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        linguistic = extract_linguistic_features(text)

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "linguistic_features": torch.tensor(
                linguistic,
                dtype=torch.float32,
            ),
            "labels": torch.tensor(
                label,
                dtype=torch.long,
            ),
        }