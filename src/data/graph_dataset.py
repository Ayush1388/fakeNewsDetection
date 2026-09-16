from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .features import extract_linguistic_features
from .propagation import load_propagation_tree


class PropagationDataset(Dataset):
    """
    Dataset for propagation-aware fake-news detection.

    Expected dataframe columns:
        id
        text
        label

    Constructor order matches train_propagation.py:
        PropagationDataset(
            dataframe,
            tokenizer,
            tree_dir,
            max_length,
        )
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        tree_dir: str | Path,
        max_length: int = 256,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.tree_dir = Path(tree_dir)
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        text = str(row["text"])
        tweet_id = str(row["id"])
        label = int(row["label"])

        # Source tweet tokenization
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Linguistic features
        linguistic_features = torch.tensor(
            extract_linguistic_features(text),
            dtype=torch.float32,
        )

        # Propagation tree
        tree = load_propagation_tree(
            self.tree_dir,
            tweet_id,
        )

        node_features = torch.tensor(
            tree["node_features"],
            dtype=torch.float32,
        )

        edge_index = torch.tensor(
            tree["edge_index"],
            dtype=torch.long,
        )

        delays = torch.tensor(
            tree["delays"],
            dtype=torch.float32,
        )

        return {
            "id": tweet_id,
            "text": text,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "linguistic_features": linguistic_features,
            "node_features": node_features,
            "edge_index": edge_index,
            "delays": delays,
            "label": torch.tensor(
                label,
                dtype=torch.long,
            ),
        }


def propagation_collate(batch: List[Dict]) -> Dict:
    """
    Collate function for variable-size propagation graphs.
    """

    return {
        "ids": [item["id"] for item in batch],
        "texts": [item["text"] for item in batch],

        "input_ids": torch.stack(
            [item["input_ids"] for item in batch]
        ),

        "attention_mask": torch.stack(
            [item["attention_mask"] for item in batch]
        ),

        "linguistic_features": torch.stack(
            [item["linguistic_features"] for item in batch]
        ),

        # Graphs have variable node/edge counts.
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
            [item["label"] for item in batch]
        ),
    }


# Backwards-compatible alias.
propagation_collate_fn = propagation_collate