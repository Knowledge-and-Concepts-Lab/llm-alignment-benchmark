##SETUP###

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

import torch
##paths/config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STIM_CONFIG_PATH = os.path.join(BASE_DIR, "..", "config", "stim_battery.json")
MODEL_CONFIG_PATH = os.path.join(BASE_DIR, "..", "config", "model_battery.json")
EXP_CONFIG_PATH = os.path.join(BASE_DIR, "..", "config", "exp_battery.json")

os.environ["TORCH_USE_CUDA_DSA"] = "1"

PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from core.hf_model_wrapper import HFModelWrapper
from core.stimuli_set import StimuliSet
from core.spose_embeddings import SposeEmbeddings
import utils.embeddings_utils as embeddings_utils 

import spose.train as spose

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

MODEL_JSON = load_json(MODEL_CONFIG_PATH)
STIM_JSON = load_json(STIM_CONFIG_PATH)
EXP_JSON = load_json(EXP_CONFIG_PATH)


###EXPERIMENTS###

def triplet_run_1_a(model_config, stimuli_key, **kwargs):
    """
    Run triplet inference experiment 1a
    """
    global experiment_name

    model = MODEL_JSON[model_config]
    stimuli = STIM_JSON[stimuli_key]
    exp = EXP_JSON[experiment_name]

    hf_model = HFModelWrapper(
        model["model_name"],
        tokenizer=model["tokenizer_name"],
        do_chat_template=model["do_chat_template"],
        model_load=model["model_load"],
        cache_dir=os.getenv("CACHE_DIR")
    )

    things_stimuli = StimuliSet(
        stimuli["items_tsv"],
        stimuli["src_key"],
        exp["instruct_prompt"],
        n_items=exp["n_items"],
        ref_embeddings=stimuli["human_embeddings"],
        euclid_min=exp["euclid_min"],
        euclid_max=exp["euclid_max"]
    )

    things_df = things_stimuli.get_stimuli_df()
    model_res_list = hf_model.do_model_batch_generation(
        things_df["format_str"],
        max_new_tokens=25,
        instr_prompt=model["answer_format"],
        triplet_cols=[
            things_df["item_x"],
            things_df["item_y"],
            things_df["item_z"]
        ]
    )

    # check for optional version_dir
    version_dir = kwargs.get("version_dir", None)
    if version_dir:
        raw_out_dir = os.path.join("results", version_dir, "raw_out", model_config)
        embed_out_dir = os.path.join("results", version_dir, "embed_in", model_config)
        os.makedirs(raw_out_dir, exist_ok=True)
        os.makedirs(embed_out_dir, exist_ok=True)

        output_csv = os.path.join(raw_out_dir, f"model_triplet_output_{model_config}_{stimuli_key}.csv")
        train_path = os.path.join(embed_out_dir, "train_90.txt")
        test_path = os.path.join(embed_out_dir, "test_10.txt")
    else:
        output_csv = f"model_triplet_output_{model_config}_{stimuli_key}.csv"
        train_path = f"train_90_{model_config}_{stimuli_key}.txt"
        test_path = f"test_10_{model_config}_{stimuli_key}.txt"

    # Save model output CSV
    pd.DataFrame(model_res_list).to_csv(output_csv, index=False)
    print(f"[INFO] Saved model triplet results to {output_csv}")

    # Integer triplet mapping
    integer_mappings = things_stimuli.get_item_index_mapping(stimuli["src_key"])
    raw_index_triplets = embeddings_utils.map_triplet_dicts_to_indices(integer_mappings, model_res_list)
    raw_index_triplets = np.array(raw_index_triplets)
    n_total = raw_index_triplets.shape[0]
    # Split into train/test
    n_test = max(1, int(n_total * 0.1))
    test_indices = np.random.choice(n_total, size=n_test, replace=False)
    train_indices = np.setdiff1d(np.arange(n_total), test_indices)

    np.savetxt(train_path, raw_index_triplets[train_indices], fmt="%d")
    np.savetxt(test_path, raw_index_triplets[test_indices], fmt="%d")

    print(f"[INFO] Saved train triplets to {train_path}")
    print(f"[INFO] Saved test triplets to {test_path}")



def embedding_1_a(model_config, stimuli_key, **kwargs):
    """
    Create embeddings from model responses
    """
    global experiment_name

    embeddings_params = EXP_JSON[experiment_name]

    # Require version_dir
    version_dir = kwargs.get("version_dir", None)
    if version_dir is None:
        raise ValueError("embedding_1_a requires a 'version_dir' argument")

    # Compose input/output paths
    triplets_dir = os.path.join("results", version_dir, "embed_in", model_config, "train_90.txt")
    results_dir = os.path.join("results", version_dir, "embed_out", model_config)
    plots_dir = results_dir  # Use same dir for plots

    if not os.path.exists(triplets_dir):
        raise FileNotFoundError(f"[ERROR] Could not find triplets file at {triplets_dir}")
    os.makedirs(results_dir, exist_ok=True)

    print(f"[INFO] Using triplets file: {triplets_dir}")
    print(f"[INFO] Saving SPoSE outputs to: {results_dir}")

    # Compose full args dict from config
    spose_args = {
        "task": "odd_one_out",
        "rnd_seed": embeddings_params.get("rnd_seed", 42),
        "modality": embeddings_params.get("modality", "behavioral/"),
        "results_dir": results_dir,
        "plots_dir": plots_dir,
        "triplets_dir": triplets_dir,
        "device": torch.device("cuda:0"),
        "batch_size": embeddings_params.get("batch_size", 100),
        "embed_dim": embeddings_params.get("embed_dim", 90),
        "epochs": embeddings_params.get("epochs", 500),
        "window_size": embeddings_params.get("window_size", 50),
        "sampling_method": embeddings_params.get("sampling_method", "normal"),
        "lmbda": embeddings_params.get("lmbda", 0.001),
        "lr": embeddings_params.get("lr", 0.001),
        "steps": embeddings_params.get("steps", 10),
        "resume": embeddings_params.get("resume", False),
        "p": embeddings_params.get("p", None),
        "distance_metric": embeddings_params.get("distance_metric", "dot"),
        "temperature": embeddings_params.get("temperature", 1.0),
        "early_stopping": embeddings_params.get("early_stopping", True)
    }

    # Call SPoSE training
    spose.run(**spose_args)
    print(f"[INFO] SPoSE embedding completed. Results saved to {results_dir}")



EXPERIMENTS = {
    "triplet_run_1_a": triplet_run_1_a,
    "embedding_1_a": embedding_1_a
}

###CMDLINE ARGS####
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python script/exp_battery.py <experiment_name> <model_config_name> <stimuli_key> [key=value] ...")
        sys.exit(1)

    experiment_name = sys.argv[1]
    model_config_name = sys.argv[2]
    stimuli_key = sys.argv[3]

    extra_kwargs = {}
    for arg in sys.argv[4:]:
        if "=" in arg:
            key, value = arg.split("=", 1)
            extra_kwargs[key] = value
        else:
            print(f"Warning: Ignored argument '{arg}' (not in key=value format)")

    exp_input = None
    try:
        with open(EXP_CONFIG_PATH, "r") as f:
            exp_config = json.load(f)
            if experiment_name in exp_config:
                exp_input = exp_config[experiment_name]
                print(f"Loaded exp_input for experiment '{experiment_name}' from {EXP_CONFIG_PATH}.")
            else:
                print(f"No exp_input found for experiment '{experiment_name}' in {EXP_CONFIG_PATH}. Proceeding without.")
    except Exception as e:
        print(f"Warning: Could not load {EXP_CONFIG_PATH}. Proceeding without exp_input. Error: {e}")


    if experiment_name in EXPERIMENTS:
        EXPERIMENTS[experiment_name](model_config_name, stimuli_key, **extra_kwargs)
    else:
        print(f"Error: Experiment '{experiment_name}' not found.")
        sys.exit(1)