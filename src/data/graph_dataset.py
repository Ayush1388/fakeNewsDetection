from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .features import extract_linguistic_features
from .propagation import load_propagation_tree


class PropagationDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        tree_root,
        max_length=256,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.tree_root = Path(tree_root)
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        text = str(row["text"])
        tweet_id = str(row["tweet_id"])

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        graph = load_propagation_tree(
            self.tree_root,
            tweet_id,
        )

        linguistic = extract_linguistic_features(
            text
        )

        return {
            "input_ids": encoded[
                "input_ids"
            ].squeeze(0),

            "attention_mask": encoded[
                "attention_mask"
            ].squeeze(0),

            "linguistic_features": torch.tensor(
                linguistic,
                dtype=torch.float32,
            ),

            "node_features": torch.tensor(
                graph["node_features"],
                dtype=torch.float32,
            ),

            "edge_index": torch.tensor(
                graph["edge_index"],
                dtype=torch.long,
            ),

            "delays": torch.tensor(
                graph["delays"],
                dtype=torch.float32,
            ),

            "labels": torch.tensor(
                int(row["label"]),
                dtype=torch.long,
            ),

            "tweet_id": tweet_id,
        }


def propagation_collate(batch):

    return {
        "input_ids": torch.stack(
            [
                item["input_ids"]
                for item in batch
            ]
        ),

        "attention_mask": torch.stack(
            [
                item["attention_mask"]
                for item in batch
            ]
        ),

        "linguistic_features": torch.stack(
            [
                item["linguistic_features"]
                for item in batch
            ]
        ),

        # Variable-size graphs.
        "node_features": [
            item["node_features"]
            for item in batch
        ],

        "edge_index": [
            item["edge_index"]
            for item in batch
        ],

        "delays": [
            item["delays"]
            for item in batch
        ],

        "labels": torch.stack(
            [
                item["labels"]
                for item in batch
            ]
        ),

        "tweet_id": [
            item["tweet_id"]
            for item in batch
        ],
    }