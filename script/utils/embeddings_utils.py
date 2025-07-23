from torch import Tensor, nn
import gc
import os
import sys
import json
import torch


def map_triplet_dicts_to_indices(item2idx: dict, triplet_dicts: list[dict]) -> list[list[int]]:
    """
    Map triplet dictionaries to integer values based on item2idx mapping., from original stimuli set.
    """
    indexed_triplets = []

    for row in triplet_dicts:
        try:
            triplet = [item2idx[row[k]] for k in row]
            indexed_triplets.append(triplet)
        except KeyError as e:
            raise KeyError(f"Value '{e.args[0]}' in row {row} not found in item2idx mapping.")

    return indexed_triplets
