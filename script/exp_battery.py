##SETUP###

from torch import Tensor, nn

from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from tqdm.auto import tqdm
from transformer_lens import ActivationCache, HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from tqdm import tqdm
import numpy as np
import gc
import requests
import os
import sys
import json

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

    model = HFModelWrapper(
            model["model_name"],
            tokenizer=model["tokenizer_name"],
            do_chat_template=model["do_chat_template"],
            model_load=model["model_load"],
            cache_dir=os.getenv("CACHE_DIR")
        )
    
    things_stimuli = StimuliSet(stimuli["items_tsv"], stimuli["src_key"], exp["instruct_prompt"], ref_embeddings=stimuli["human_embeddings"])

    things_stimuli.export_embeddings_csv("embeddings_human_out.csv")

    


EXPERIMENTS = {
    "triplet_run_1_a": triplet_run_1_a
}




###CMDLINE ARGS####
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script/exp_battery.py <experiment_name> <model_config_name> <stimuli_key>")
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