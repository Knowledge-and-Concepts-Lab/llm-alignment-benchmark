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

# --- SETUP FLEXIBLE PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")  
sys.path.append(PROJECT_ROOT) 

os.environ["HF_HOME"] = "/mnt/dv/wid/projects3/Rogers-muri-human-ai/zstuddiford"
os.environ["XDG_CACHE_HOME"] = "/mnt/dv/wid/projects3/Rogers-muri-human-ai/zstuddiford"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/dv/wid/projects3/Rogers-muri-human-ai/zstuddiford"
os.environ["HF_HUB_CACHE"] = "/mnt/dv/wid/projects3/Rogers-muri-human-ai/zstuddiford"



class HFModelWrapper:
    def __init__(self, model_name:str, tokenizer:str=None, do_chat_template:bool=False, device:str="cuda", cache_dir:str=None, model_load="direct"):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.tokenizer = self.load_tokenizer(tokenizer) if tokenizer != None else None

        self.model = self.load_model_direct(model_name) if model_load == "direct" else self.load_model_pipeline(model_name)
        self.model_load = model_load

        self.do_chat_template = do_chat_template
        


    def load_tokenizer(self, tokenizer_str):
        """
        Load the Hugging Face tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_str, 
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )

        return tokenizer
        
    def load_model_pipeline(self, model_str):
        """
        Load the Hugging Face model.
        """

        pipe = pipeline("text-generation", 
            model=model_str, 
            device="cuda", 
            torch_dtype=torch.bfloat16,
            tokenizer=self.tokenizer,
            trust_remote_code=True
        )
        return pipe


    def load_model_direct(self, model_str):
        """
        Load the Hugging Face model directly.
        """

        model = AutoModelForCausalLM.from_pretrained(
            model_str, 
            cache_dir=self.cache_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        return model


    def do_model_generation(self, prompt: str, 
                            max_new_tokens: int = 10, 
                            do_sample: bool = True):
        """
        Generate text using the model.
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Please provide a model.")
            
        if self.model_load == "pipeline":
            output = self.model(prompt)
            return output[0]['generated_text']

        elif self.model_load == "direct":

            #check for prompt template and format tokens accordingly
            if self.do_chat_template:
                if isinstance(prompt, str):
                    prompt = [{"role": "user", "content": prompt}]
                input_ids = self.tokenizer.apply_chat_template(
                    prompt, 
                    tokenize=True, 
                    return_tensors="pt", 
                    add_generation_prompt=True
                ).to(self.model.device)
                print(self.tokenizer.decode(input_ids[0]))


            else:
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()

            generated_ids = self.model.generate(input_ids, max_new_tokens=max_new_tokens)
            return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


if __name__ == "__main__":

    model_name = "google/gemma-3-1b-it"

    tokenizer_name = model_name   

    print(model_name)
    print(tokenizer_name) 
    wrapper = HFModelWrapper(
        model_name, 
        tokenizer=tokenizer_name, 
        do_chat_template=True,
        cache_dir="/mnt/dv/wid/projects3/Rogers-muri-human-ai/zstuddiford", 
        model_load="direct"
    )
    
    prompt = "What is the capital of France?"

    response = wrapper.do_model_generation(prompt, max_new_tokens=10)

    print(response)
    
    