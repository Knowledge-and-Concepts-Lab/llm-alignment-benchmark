from torch import Tensor, nn

from tqdm.auto import tqdm
from tqdm import tqdm
import numpy as np
import gc
import requests
import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
from random import choice, sample
from numpy.linalg import norm

class StimuliSet:

    def __init__(self, src_tsv:str, src_key:str, format_prompt:str, n_items = 1000, ref_embeddings=None, euclid_min=0.0, euclid_max=1.0):
        
        self.src_tsv = src_tsv
        self.src_df = pd.read_csv(self.src_tsv, sep="\t")
        self.src_key=src_key

        print(f"{self.src_tsv} successfully loaded as stimuli")

        self.ref_embeddings = self.__read_embeddings(ref_embeddings, self.src_df, self.src_key) if ref_embeddings is not None else None

        self.format_prompt = format_prompt
    
        self.stimuli = self.__construct_stimuli(n_items=n_items, ref_embeddings=self.ref_embeddings, euclid_min=euclid_min, euclid_max=euclid_max)


    def __construct_stimuli(self, n_items, ref_embeddings=None, euclid_min=0.0, euclid_max=1.0) -> list[dict]:
        
        ref_embeddings=None

        stimuli = []
        keys = self.src_df[self.src_key].astype(str).tolist()

        if ref_embeddings is not None:
            print(f"[DEBUG] Using reference embeddings for distance constraints")

        attempts = 0
        max_attempts = n_items * 100  # fail-safe to prevent infinite loops

        while len(stimuli) < n_items and attempts < max_attempts:
            attempts += 1
            item_x = choice(keys)

            if ref_embeddings is None:
                # If no embedding constraints, just sample two other items
                rest = [k for k in keys if k != item_x]
                item_y, item_z = sample(rest, 2)
                dist_xy = dist_xz = dist_yz = None
            else:
                x_vec = ref_embeddings.get(item_x)
                if x_vec is None:
                    continue

                # Candidates for y (close) and z (far)
                possible_y = []
                possible_z = []

                for other in keys:
                    if other == item_x:
                        continue
                    vec = ref_embeddings.get(other)
                    if vec is None:
                        continue
                    dist = norm(x_vec - vec)
                    if dist < euclid_min:
                        possible_y.append((other, dist))
                    elif dist > euclid_max:
                        possible_z.append((other, dist))

                if not possible_y or not possible_z:
                    continue  # try again with a different x

                item_y, dist_xy = choice(possible_y)
                item_z, dist_xz = choice(possible_z)

                # Optional: compute dist_yz if needed
                y_vec = ref_embeddings[item_y]
                z_vec = ref_embeddings[item_z]
                dist_yz = norm(y_vec - z_vec)

            stimuli.append({
                "item_x": item_x,
                "item_y": item_y,
                "item_z": item_z,
                "dist_x_y": dist_xy,
                "dist_x_z": dist_xz,
                "dist_y_z": dist_yz
            })

        if attempts >= max_attempts:
            print(f"[WARN] Max attempts reached. Collected {len(stimuli)} valid triplets.")

        print(f"[DEBUG] Constructed {len(stimuli)} stimuli")
        print(stimuli)
        return stimuli

        

    
    def __read_embeddings(self, embeddings_path: str, items_df: pd.DataFrame, match_key: str) -> dict[str, np.ndarray]:
        """
        Reads a .txt embedding file where each row corresponds to an item and each column to a dimension.
        First constructs a list of {'item': ..., 'embedding': ...} entries (including duplicates),
        then builds a dict using the first occurrence of each unique item.
        """
        matrix = np.loadtxt(embeddings_path)


        if matrix.shape[0] != len(items_df):
            print(f"[WARN] Matrix row count ({matrix.shape[0]}) does not match items_df ({len(items_df)})")

        keys = items_df[match_key].astype(str).tolist()

        embedding_objects = []
        for i in range(min(len(matrix), len(keys))):
            embedding_objects.append({
                "item": keys[i],
                "embedding": matrix[i]
            })

        # Deduplicate: keep first occurrence
        unique_dict = {}
        for obj in embedding_objects:
            item = obj["item"]
            if item not in unique_dict:
                unique_dict[item] = obj["embedding"]

        print(f"Final reference embedding dict has {len(unique_dict)} unique items")

        return unique_dict



    def __get_items_in_ref_distance(self, ref_distance:float) -> list:
        """
        Get items that are within a certain distance from the reference embedding.
        """
        if self.ref_embeddings is None:
            raise ValueError("Reference embeddings not provided.")
        
        return

    def get_stimuli_list(self):
        """
        Get a list of stimuli objects for inference
        """
        return

    def get_stimuli_csv(self):
        """
        Export csv of stimuli objects
        """
        return

    def export_embeddings_csv(self, output_path: str):
        """
        Export the reference embeddings dictionary to a CSV file.
        Each row contains the item key and a stringified embedding vector.
        """
        if self.ref_embeddings is None:
            raise ValueError("Reference embeddings not loaded.")

        rows = []
        for item, vec in self.ref_embeddings.items():
            rows.append({
                "item": item,
                "embedding": " ".join(map(str, vec))
            })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"[DEBUG] Exported {len(df)} embeddings to {output_path}")


    
