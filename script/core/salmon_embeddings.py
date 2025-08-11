from torch import Tensor, nn

from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from tqdm.auto import tqdm
from transformer_lens import ActivationCache, HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from tqdm import tqdm
import numpy as np
import pandas as pd
import gc
import requests
import os
import sys
import json
from salmon.triplets.offline import OfflineEmbedding
from sklearn.model_selection import train_test_split


class SalmonEmbeddings:

    def __init__(self, csv_dir: str, config:dict, embeddings_dir: str):
        self.embeddings_dir = embeddings_dir
        self.config = config

    
    
    def create_embeddings(self, csv_dir: str):
        max_expochs = self.config.get("max_epochs", 50000)
        d = self.config.get("d", 10)

        df = pd.read_csv(csv_dir)
        keys = pd.concat([df["head"], df["winner"], df["loser"]])
        keys = keys.dropna().astype(str)
        unique_items = sorted(set(keys) - set(["-1"]))
        item_to_id = {item: idx for idx, item in enumerate(unique_items)}

        triplets = []
        for _, row in df.iterrows():
            try:
                h = item_to_id[row["head_key"]]
                w = item_to_id[row["winner_key"]]
                l = item_to_id[row["loser_key"]]
                triplets.append((h, w, l))
            except KeyError:
                continue  # skip rows with missing keys

        triplets = np.array(triplets)
        if len(triplets) == 0:
            raise ValueError("No valid triplets could be constructed from the data.")

        # Train/test split
        X_train, X_test = train_test_split(triplets, test_size=0.2, random_state=42)

        # Fit the embedding
        n = len(unique_items)
        model = OfflineEmbedding(n=n, d=d, max_epochs=max_expochs)
        model.initialize(X_train)
        model.fit(X_train, X_test)

        # Join embedding with labels
        embedding = model.embedding_
        labels = [{"item": item, "id": i} for item, i in item_to_id.items()]
        label_df = pd.DataFrame(labels).sort_values("id")
        emb_df = pd.DataFrame(embedding, columns=[f"dim_{i}" for i in range(d)])
        result_df = pd.concat([label_df.reset_index(drop=True), emb_df], axis=1)

        # Save
        result_df.to_csv(self.embeddings_dir, index=False)
        print(f"Saved embedding to {self.embeddings_dir}")
            