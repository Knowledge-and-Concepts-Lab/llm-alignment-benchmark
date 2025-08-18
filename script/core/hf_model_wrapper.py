from __future__ import annotations

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
from utils.performance_utils import timeit

import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import utils.model_utils as model_utils


class HFModelWrapper:
    def __init__(
        self,
        model_name: str,
        tokenizer: str | None = None,
        do_chat_template: bool = False,
        device: str = "cuda",
        cache_dir: str | None = None,
        model_load: str = "direct",
        experiment_name: str | None = None,
    ):
        """
        Args:
            model_name: HF repo id or local path.
            tokenizer: Optional tokenizer repo id (defaults to model's tokenizer if None).
            do_chat_template: If True, use tokenizer.apply_chat_template for prompts.
            device: cuda or cpu
            cache_dir: HF cache directory.
            model_load: "direct" or "pipeline"- (use direct unless otherwise specified).
            experiment_name: Optional experiment label for runtime logs.
                             If None, will use $EXPERIMENT_NAME or 'default_exp'.
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_load = model_load
        self.do_chat_template = do_chat_template

        # experiment label for runtime logs
        self.experiment_name = (
            experiment_name
            if experiment_name is not None
            else os.getenv("EXPERIMENT_NAME", "default_exp")
        )

        # Load tokenizer first (some codepaths need it during generation)
        self.tokenizer = self.load_tokenizer(tokenizer) if tokenizer is not None else None

        # Load model
        self.model = (
            self.load_model_direct(model_name)
            if model_load == "direct"
            else self.load_model_pipeline(model_name)
        )


        # ------ runtime logging setup ---------
        log_path = self._runtime_log_path()

        self.do_model_generation = timeit(
            log_path=log_path, label="do_model_generation"
        )(self.do_model_generation)
        self.do_model_batch_generation = timeit(
            log_path=log_path, label="do_model_batch_generation"
        )(self.do_model_batch_generation)


    def load_tokenizer(self, tokenizer_str):
        """
        Load HF tokenizer.
        Args:
            tokenizer_str: Hugging Face repo id or local path for the tokenizer.
        Returns:
            AutoTokenizer instance.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_str,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )
        return tokenizer

    def load_model_pipeline(self, model_str):
        """
        Load the Hugging Face model using a pipeline. Use this if loading model direct doesn't work.
        Args:
            model_str: Hugging Face repo id or local path for the model.
        Returns:
            A text-generation pipeline instance.
        """
        pipe = pipeline(
            "text-generation",
            model=model_str,
            device="cuda",
            torch_dtype=torch.bfloat16,
            tokenizer=self.tokenizer,
            trust_remote_code=True,
        )
        return pipe

    def load_model_direct(self, model_str):
        """
        Load the Hugging Face model directly.
        Args:
            model_str: Hugging Face repo id or local path for the model.
        Returns:
            An AutoModelForCausalLM instance.
        """

        model = AutoModelForCausalLM.from_pretrained(
            model_str,
            cache_dir=self.cache_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        return model


    def do_model_generation(
        self,
        prompt: str,
        max_new_tokens: int = 10,
        instr_prompt: str | None = None,
    ) -> str:
        """
        Generate text from the model given a prompt.
        
        Args:
            prompt: The input text prompt to generate from.
            max_new_tokens: Maximum number of tokens to generate.
            instr_prompt: Optional instruction prompt to append (e.g., "answer:").
        Returns:
            Generated text as a string.
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Please provide a model.")

        if self.model_load == "pipeline":
            output = self.model(prompt)
            return output[0]["generated_text"]

        elif self.model_load == "direct":
            # check for prompt template and format tokens accordingly
            if self.do_chat_template:
                if isinstance(prompt, str):
                    prompt = [{"role": "user", "content": prompt}]
                input_ids = self.tokenizer.apply_chat_template(
                    prompt,
                    tokenize=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                ).to(self.model.device)
            else:
                if instr_prompt:  # add instr prompt (e.g., "answer:") if specified
                    prompt = prompt + instr_prompt
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()

            generated_ids = self.model.generate(input_ids, max_new_tokens=max_new_tokens)
            return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    @torch.inference_mode()
    def do_model_batch_generation(
        self,
        prompt_list: List[str],
        max_new_tokens: int,
        instr_prompt: str | None,
        triplet_cols: List[List[str]],
        batch_size: int = 6,
        pad_to_multiple_of: int = 8,
        use_length_bucketing: bool = True,
        choose_mode: Literal["tokens", "logits"] = "tokens",
    ):
        """
        Run model batch inference for batch_size n.
        Args:
            prompt_list: List of prompts to generate from.
            max_new_tokens: Maximum number of tokens to generate.
            instr_prompt: Optional instruction prompt to append (e.g., "answer:").
            triplet_cols: List of three lists [x, y, z] where:
                - x: List of heads (e.g., "head1", "head2", ...).
                - y: List of first options (e.g., "option1", "option2", ...).
                - z: List of second options (e.g., "option3", "option4", ...).
            batch_size: Number of prompts to process in a single batch.
            pad_to_multiple_of: Padding size for tokenization.
            use_length_bucketing: If True, sort prompts by length before batching.
            choose_mode: "tokens" for generation mode, "logits" for logits comparison.
        Returns:
            List of results objects with each entry containing:
                - head: The head from triplet_cols.
                - winner: The chosen option (y or z).
                - loser: The other option (z or y).
        """
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
            print(f"BATCH IDX {batch_idx}")
            batch_prompts_raw = [prompt_list[i] for i in batch_idx]
            batch_heads = [x[i] for i in batch_idx]
            batch_y = [y[i] for i in batch_idx]
            batch_z = [z[i] for i in batch_idx]

            texts = self._build_prompt_strings(batch_prompts_raw, instr_prompt)

            if choose_mode == "logits":
                winners, losers, lp_y, lp_z, y_ids, z_ids = self._choose_logits_token(
                    texts,
                    batch_y,
                    batch_z,
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

            # ===== generation mode =====
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
            cleaned = [
                t[len(p):].strip() if t.startswith(p) else t.strip()
                for t, p in zip(decoded, texts)
            ]
            for j, res_text in enumerate(cleaned):
                chosen = model_utils.clean_model_response_match(
                    res_text, [batch_y[j], batch_z[j]]
                )
                if chosen == batch_y[j]:
                    winner, loser = batch_y[j], batch_z[j]
                elif chosen == batch_z[j]:
                    winner, loser = batch_z[j], batch_y[j]
                else:
                    winner = loser = None
                results[batch_idx[j]] = {
                    "head": batch_heads[j],
                    "winner": winner,
                    "loser": loser,
                    "raw_generation": cleaned[j],
                }

        for start in range(0, n, batch_size):
            run_batch(indices[start : start + batch_size])

        return results

    # ---------- utilities ----------

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

    def _first_token_ids(self, texts: List[str]) -> List[int]:
        """
        Tokenize each string and return the ID of its first emitted token.
        Uses right padding so index 0 is the first token.
        """
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
    def _choose_logits_token(
        self,
        prompt_texts: List[str],
        y_list: List[str],
        z_list: List[str],
        pad_to_multiple_of: int = 8,
    ):
        """
        Compare log P(first_token(y) | prompt) vs log P(first_token(z) | prompt).
        Returns (winner_list, loser_list, lp_y, lp_z, y_token_ids, z_token_ids).

        Args:
            prompt_texts: List of prompt strings to evaluate.
            y_list: List of first options (y).
            z_list: List of second options (z).
            pad_to_multiple_of: Padding size for tokenization.
        Returns:
            - winner_list: List of winning options (y or z) for each prompt.
            - loser_list: List of losing options (z or y) for each prompt.
            - lp_y: Log probabilities for y.
            - lp_z: Log probabilities for z.
            - y_token_ids: Token IDs for y.
            - z_token_ids: Token IDs for z.
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
        last_idx = attn_mask.sum(dim=1) - 1
        row = torch.arange(input_ids.size(0), device=device)
        next_logits = logits[row, last_idx]  # (B, V)
        logprobs = F.log_softmax(next_logits, dim=-1)  # (B, V)

        # First token ids for y and z
        y_ids = torch.tensor(self._first_token_ids(y_list), device=device)
        z_ids = torch.tensor(self._first_token_ids(z_list), device=device)

        lp_y = logprobs[row, y_ids]
        lp_z = logprobs[row, z_ids]

        winners, losers = [], []
        for i in range(len(y_list)):
            if torch.isnan(lp_y[i]) or torch.isnan(lp_z[i]):
                winners.append(None)
                losers.append(None)
            elif lp_y[i] > lp_z[i]:
                winners.append(y_list[i])
                losers.append(z_list[i])
            elif lp_z[i] > lp_y[i]:
                winners.append(z_list[i])
                losers.append(y_list[i])
            else:
                winners.append(None)
                losers.append(None)

        return winners, losers, lp_y.tolist(), lp_z.tolist(), y_ids.tolist(), z_ids.tolist()
    
    
    # ---------- helpers for runtime logging ----------

    def _runtime_log_path(self) -> str:
        """
        Create path for exporting runtime info txt
        """
        safe_model = str(self.model_name).replace("/", "_").replace(" ", "_")
        safe_exp = str(self.experiment_name).replace("/", "_").replace(" ", "_")
        path = os.path.join("results", f"runtime_{safe_model}_{safe_exp}.txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def set_experiment_name(self, experiment_name: str) -> None:
        """
        Set experiment name for logging.
        """
        self.experiment_name = experiment_name
        log_path = self._runtime_log_path()
        # re-wrap with new log path
        self.do_model_generation = timeit(
            log_path=log_path, label="do_model_generation"
        )(self._unwrap(self.do_model_generation))
        self.do_model_batch_generation = timeit(
            log_path=log_path, label="do_model_batch_generation"
        )(self._unwrap(self.do_model_batch_generation))

    @staticmethod
    def _unwrap(fn):
        """
        Best-effort unwrapping if a function has been wrapped by decorators.
        Falls back to the function itself if no __wrapped__ present.
        """
        return getattr(fn, "__wrapped__", fn)
