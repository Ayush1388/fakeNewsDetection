from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizerBase

from .features import extract_linguistic_features
from .propagation import load_propagation_tree


class PropagationDataset(Dataset):
    """
    Dataset for fake-news detection using:

        source tweet text
        linguistic features
        propagation tree structure
        propagation delays

    Expected dataframe columns:

        id
        text
        label

    Each propagation tree is loaded using the source tweet ID.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tree_dir: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tree_dir = Path(tree_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Dict:
        row = self.dataframe.iloc[idx]

        # -------------------------------------------------
        # Source tweet
        # -------------------------------------------------

        tweet_id = str(row["id"])
        text = str(row["text"])
        label = int(row["label"])

        # -------------------------------------------------
        # Tokenize source tweet
        # -------------------------------------------------

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # -------------------------------------------------
        # Linguistic features
        # -------------------------------------------------

        linguistic_features = torch.tensor(
            extract_linguistic_features(text),
            dtype=torch.float32,
        )

        # -------------------------------------------------
        # Propagation tree
        # -------------------------------------------------

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
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "linguistic_features": linguistic_features,
            "node_features": node_features,
            "edge_index": edge_index,
            "delays": delays,
            "label": torch.tensor(label, dtype=torch.long),
        }


def propagation_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for variable-size propagation trees.

    Text tensors have fixed dimensions and can be stacked.

    Graph tensors cannot be stacked because every propagation
    tree has a different number of nodes and edges.

    Therefore graph-related fields remain Python lists.
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

        "node_features": [
            item["node_features"] for item in batch
        ],

        "edge_index": [
            item["edge_index"] for item in batch
        ],

        "delays": [
            item["delays"] for item in batch
        ],

        "labels": torch.stack(
            [item["label"] for item in batch]
        ),
    }