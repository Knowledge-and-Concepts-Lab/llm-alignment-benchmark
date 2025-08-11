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

import math
from typing import List, Tuple, Literal
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch.nn.functional as F

import utils.model_utils as model_utils


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
                            instr_prompt: str=None):
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
                if instr_prompt: #add instr prompt (i.e "answer:") if specified
                    prompt = prompt + instr_prompt
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()

            generated_ids = self.model.generate(input_ids, max_new_tokens=max_new_tokens)
            return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        

    def _build_prompt_strings(self, prompts: List[str], instr_prompt: str | None):
        """
        Returns a list of prompt strings ready to tokenize.
        Handles chat template mode by turning each prompt into a chat.
        """
        if self.do_chat_template:
            # Build raw chat strings via the tokenizer's chat template
            rendered = []
            for p in prompts:
                chat = [{"role": "user", "content": p if instr_prompt is None else f"{p}{instr_prompt}"}]
                text = self.tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
                rendered.append(text)
            return rendered
        else:
            if instr_prompt:
                return [p + instr_prompt for p in prompts]
            return prompts


    @torch.inference_mode()
    def do_model_batch_generation(
        self,
        prompt_list: List[str],
        max_new_tokens: int,
        instr_prompt: str | None,
        triplet_cols: List[List[str]],
        batch_size: int = 200,
        pad_to_multiple_of: int = 8,
        use_length_bucketing: bool = True,
        choose_mode: Literal["hard", "soft_first"] = "hard",
    ):

        if self.model is None or self.tokenizer is None:
            raise ValueError("Model/tokenizer not loaded.")
        if triplet_cols is None or len(triplet_cols) != 3:
            raise ValueError("triplet_cols must be [x, y, z].")
        if not all(len(col) == len(prompt_list) for col in triplet_cols):
            raise ValueError("Each triplet column must match prompt_list length.")

        self.model.eval()
        device = self.model.device
        x, y, z = triplet_cols
        n = len(prompt_list)
        indices = list(range(n))

        if use_length_bucketing:
            lengths = [len(p) for p in prompt_list]
            indices = sorted(indices, key=lambda i: lengths[i])

        results = [None] * n

        def run_batch(batch_idx: List[int]):
            batch_prompts_raw = [prompt_list[i] for i in batch_idx]
            batch_heads = [x[i] for i in batch_idx]
            batch_y = [y[i] for i in batch_idx]
            batch_z = [z[i] for i in batch_idx]

            texts = self._build_prompt_strings(batch_prompts_raw, instr_prompt)

            if choose_mode == "soft_first":
                winners, losers, lp_y, lp_z, y_ids, z_ids = self._choose_soft_first_token(
                    texts, batch_y, batch_z,
                    pad_to_multiple_of=pad_to_multiple_of,
                )
                for j in range(len(batch_idx)):
                    results[batch_idx[j]] = {
                        "head": batch_heads[j],
                        "winner": winners[j],
                        "loser": losers[j],
                        "lp_y1": lp_y[j],
                        "lp_z1": lp_z[j],
                        "y_tok": y_ids[j],
                        "z_tok": z_ids[j],
                    }
                return

            # ===== HARD MODE (unchanged) =====
            toks = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                pad_to_multiple_of=pad_to_multiple_of,
            )
            toks = {k: v.to(device) for k, v in toks.items()}
            out_ids = self.model.generate(**toks, max_new_tokens=max_new_tokens)
            decoded = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            cleaned = [t[len(p):].strip() if t.startswith(p) else t.strip() for t, p in zip(decoded, texts)]
            for j, res_text in enumerate(cleaned):
                chosen = model_utils.clean_model_response_match(res_text, [batch_y[j], batch_z[j]])
                if chosen == batch_y[j]:
                    winner, loser = batch_y[j], batch_z[j]
                elif chosen == batch_z[j]:
                    winner, loser = batch_z[j], batch_y[j]
                else:
                    winner = loser = None
                results[batch_idx[j]] = {"head": batch_heads[j], "winner": winner, "loser": loser, "raw_generation": cleaned[j]}

        for start in range(0, n, batch_size):
            run_batch(indices[start:start + batch_size])

        return results

    def _first_token_ids(
        self,
        texts: List[str],
    ) -> List[int]:
        """
        Tokenize each string and return the ID of its first emitted token.
        Uses right padding so index 0 is the first token.
        """
        # (Optionally) add a leading space to match GPT-style merges.
        items = [t for t in texts]

        old_side = self.tokenizer.padding_side
        try:
            self.tokenizer.padding_side = "right"
            enc = self.tokenizer(
                items,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
            )
        finally:
            self.tokenizer.padding_side = old_side

        ids = enc["input_ids"]  # (B, T)
        return [int(row[0].item()) if row.numel() > 0 else self.tokenizer.eos_token_id for row in ids]
    
    @torch.inference_mode()
    def _choose_soft_first_token(
        self,
        prompt_texts: List[str],
        y_list: List[str],
        z_list: List[str],
        pad_to_multiple_of: int = 8,
    ):
        """
        Compare log P(first_token(y) | prompt) vs log P(first_token(z) | prompt).
        Returns (winner_list, loser_list, lp_y, lp_z, y_token_ids, z_token_ids).
        """
        device = self.model.device

        # Tokenize prompts (already rendered via chat template if enabled)
        enc = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            pad_to_multiple_of=pad_to_multiple_of,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        # Forward pass (no generation)
        out = self.model(input_ids=input_ids, attention_mask=attn_mask)
        logits = out.logits.float()  # (B, T, V)

        # Next-token position per row = last non-pad token
        last_idx = attn_mask.sum(dim=1) - 1                         # (B,)
        row = torch.arange(input_ids.size(0), device=device)        # (B,)
        next_logits = logits[row, last_idx]                         # (B, V)
        logprobs = F.log_softmax(next_logits, dim=-1)               # (B, V)

        # First token ids for y and z
        y_ids = torch.tensor(self._first_token_ids(y_list), device=device)
        z_ids = torch.tensor(self._first_token_ids(z_list), device=device)

        lp_y = logprobs[row, y_ids]
        lp_z = logprobs[row, z_ids]

        winners, losers = [], []
        for i in range(len(y_list)):
            if torch.isnan(lp_y[i]) or torch.isnan(lp_z[i]):
                winners.append(None); losers.append(None)
            elif lp_y[i] > lp_z[i]:
                winners.append(y_list[i]); losers.append(z_list[i])
            elif lp_z[i] > lp_y[i]:
                winners.append(z_list[i]); losers.append(y_list[i])
            else:
                winners.append(None); losers.append(None)

        return winners, losers, lp_y.tolist(), lp_z.tolist(), y_ids.tolist(), z_ids.tolist()


        