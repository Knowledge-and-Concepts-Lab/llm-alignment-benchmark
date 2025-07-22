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


class StimuliSet:

    def __init__(self, src_csv:str, format_prompt:str, n_items = 1000, ref_embeddings=None, euclid_min=0.0, euclid_max=1.0):
        
        self.items_df = pd.read_csv(src_csv)
        self.ref_embeddings = ref_embeddings
        self.format_prompt = format_prompt

        self.stimuli = self.__construct_stimuli(n_items=n_items, ref_embeddings=self.ref_embeddings, euclid_min=euclid_min, euclid_max=euclid_max)


    def __construct_stimuli(self, n_items, ref_embeddings=False, euclid_min=0.0, euclid_max=1.0) -> dict:

        return


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

    def get_stimuli_csv(self):
        """
        Export csv of stimuli objects
        """

    
