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
        """
        Initialize the SalmonEmbeddings class.
        
        Args:
            csv_dir: Path to the CSV file containing triplet data.
            config: Configuration dictionary with parameters for embedding creation.
            embeddings_dir: Directory where the resulting embeddings will be saved.
        """
        self.csv_dir = csv_dir
        self.embeddings_dir = embeddings_dir
        self.config = config

    
    def create_embeddings(self, csv_dir: str):
        """
        Create embeddings from triplet data in the specified CSV file.
        
        Args:
            csv_dir: Path to the CSV file containing triplet data.
        """
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
                h = item_to_id[row["head"]]
                w = item_to_id[row["winner"]]
                l = item_to_id[row["loser"]]
                triplets.append((h, w, l))
            except KeyError:
                continue  # skip rows with missing keys

        triplets = np.array(triplets)

        print(triplets)
        if len(triplets) == 0:
            raise ValueError("No valid triplets could be constructed from the data.")
        
        # Train/test split
        X_train, X_test = train_test_split(triplets, test_size=0.2, random_state=42)

        print(X_train.shape, X_test.shape)

        # Fit the embedding
        n = len(unique_items)
        print("this is n:", n)

        model = OfflineEmbedding(n=n, d=d, max_epochs=max_expochs, verbose=1)
        model.initialize(X_train)
        model.fit(X_train, X_test, verbose=1)

        print("we made it")

        # Join embedding with labels
        embedding = model.embedding_
        labels = [{"item": item, "id": i} for item, i in item_to_id.items()]
        label_df = pd.DataFrame(labels).sort_values("id")
        emb_df = pd.DataFrame(embedding, columns=[f"dim_{i}" for i in range(d)])
        result_df = pd.concat([label_df.reset_index(drop=True), emb_df], axis=1)

        # Save
        result_df.to_csv(self.embeddings_dir, index=False)
        print(f"Saved embedding to {self.embeddings_dir}")
            