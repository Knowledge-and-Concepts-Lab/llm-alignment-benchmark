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
from transformers import AutoTokenizer, pipeline



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
    def __init__(self, model_name:str, tokenizer:str=None, device:str="cuda", cache_dir:str=None):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.tokenizer = self.load_tokenizer(tokenizer) if tokenizer != None else None

        self.model = self.load_model_pipeline(model_name)
        self.template_str = None
        

    def set_template(self, template_str: str):
        """
        Set the prompt template string for the model.
        """
        self.template_str = template_str

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

        print("HF_HOME:", os.environ.get("HF_HOME"))
        print("XDG_CACHE_HOME:", os.environ.get("XDG_CACHE_HOME"))
        print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE"))
        print("HF_HUB_CACHE:", os.environ.get("HF_HUB_CACHE"))

        pipe = pipeline("text-generation", 
            model=model_str, 
            device="cuda", 
            torch_dtype=torch.bfloat16,
            tokenizer=self.tokenizer,
            trust_remote_code=True
        )
        return pipe
    

    def __prompt_preproc(self, prompt: str):
        """
        Preprocess the prompt string.
        """
        if self.template_str is None:
            raise ValueError("Template string is not set. Please set a template string.")
        
        # Assuming template_str is a format string
        formatted_prompt = self.template_str.format(prompt=prompt)

        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": formatted_prompt}]
            }
        ]

    def do_model_generation(self, prompt: str, 
                            max_new_tokens: int = 10, 
                            do_sample: bool = True):
        """
        Generate text using the model.
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Please provide a model.")
        
        output = self.model(prompt)

        return output[0]['generated_text']
        


if __name__ == "__main__":
    model_name = "baichuan-inc/Baichuan-7B"
    tokenizer_name = None
    
    wrapper = HFModelWrapper(model_name, tokenizer=tokenizer_name, cache_dir="/mnt/dv/wid/projects3/Rogers-muri-human-ai/zstuddiford")
    wrapper.set_template("Answer the question {prompt}")
    
    prompt = "The capital of France is"
    response = wrapper.do_model_generation(prompt, max_new_tokens=10)
    print(response)
    
        
