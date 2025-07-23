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
import pandas as pd
from random import choice, sample
from numpy.linalg import norm

import spose.train as spose

class SposeEmbeddings:

    def __init__(self, csv_dir:str, embeddings_name:str):
        self.embeddings_name = embeddings_name
        self.embeddings = self.__load_embeddings(csv_dir)

    def __load_embeddings(self, csv_dir:str):
        spose.run(
            task='odd_one_out',
            rnd_seed=42,
            modality='behavioral/',
            results_dir='./results/my_run/',
            plots_dir='./plots/my_run/',
            triplets_dir='./triplets/my_triplets.txt',  # <-- your .txt file here
            device=torch.device("cuda:0"),
            batch_size=100,
            embed_dim=90,
            epochs=500,
            window_size=50,
            sampling_method='normal',
            lmbda=0.001,
            lr=0.001,
            steps=10,
            resume=False,
            p=None,
            distance_metric='dot',
            temperature=1.0,
            early_stopping=True
        )