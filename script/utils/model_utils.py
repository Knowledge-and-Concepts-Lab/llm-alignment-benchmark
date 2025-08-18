from torch import Tensor, nn
import gc
import os
import sys
import json
import torch

def clean_model_response_match(generation_str: str, match_strs: list) -> str:
    """
    Find the first matching model response from a list of matching strings.
    """
    gen_lower = generation_str.lower()
    match_indices = {
        match: gen_lower.find(match.lower())
        for match in match_strs
    }

    # Filter out non-matches (i.e., index == -1)
    filtered = {k: v for k, v in match_indices.items() if v != -1}

    if not filtered:
        return None

    # Return the match string with the smallest index
    return min(filtered.items(), key=lambda x: x[1])[0]

