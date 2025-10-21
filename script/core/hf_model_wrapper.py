from __future__ import annotations

# ---------- stdlib ----------
from functools import wraps as _wraps
from typing import List, Tuple, Literal, Optional, Union, Sequence, Any
import atexit as _atexit
import math
import os
import sys
import json
import time
import gc
import requests

# ---------- third-party ----------
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm.auto import tqdm
from tqdm import tqdm

# PIL is optional; only needed for VLM image inputs
try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None  # We will check at runtime for VLM use

# WandB (optional auto-init helpers below use a guarded import)
import wandb

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    pipeline,
    BitsAndBytesConfig,
    # Falcon VLM (LLaVA-Next style)
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)

# ---------- local ----------
from utils.performance_utils import timeit
import utils.model_utils as model_utils
from core.models.extended_phi_4 import ExtendedPhi4FlashForCausalLM

# --- Weave wiring (optional, env-controlled) ---
import os as _os
try:
    import weave as _weave
except Exception:
    _weave = None  # graceful fallback if weave isn't installed


def _weave_enabled() -> bool:
    return (_weave is not None) and (_os.getenv("WEAVE_ENABLED", "0") == "1")


def _weave_init_if_needed():
    if not _weave_enabled():
        return
    proj = _os.getenv("WEAVE_PROJECT")
    try:
        if proj:
            _weave.init(project=proj)
        else:
            _weave.init()
    except Exception as e:
        print(f"[weave] init skipped: {e}")


def _wrap_with_weave(op_name: str, fn):
    if not _weave_enabled():
        return fn

    @_weave.op(name=op_name)
    @_wraps(fn)
    def _op(*args, **kwargs):
        return fn(*args, **kwargs)

    return _op


# ---- Optional W&B auto-init (env-controlled) ----
try:
    import wandb as _wandb
except Exception:
    _wandb = None


def _wandb_enabled() -> bool:
    return (_wandb is not None) and (_os.getenv("WANDB_ENABLED", "1") == "1")


def _wandb_auto_init(run_name: str, **config):
    if not _wandb_enabled():
        return
    try:
        if _wandb.run is None:
            _wandb.init(
                project=_os.getenv("WANDB_PROJECT", "llm-alignment-inference"),
                entity=_os.getenv("WANDB_ENTITY") or None,
                name=run_name,
                id=_os.getenv("WANDB_RUN_ID"),
                group=_os.getenv("WANDB_RUN_GROUP"),
                resume="allow",
                mode=_os.getenv("WANDB_MODE", "online"),
                config=config or None,
            )
            _atexit.register(lambda: (_wandb.run and _wandb.finish()))
    except Exception as e:
        print(f"[wandb] auto-init skipped: {e}")


# =====================================================================================
#                                    HFModelWrapper
# =====================================================================================

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
        quantization: Optional[str] = None,   # <-- NEW
    ):
        """
        quantization: choose from {"4bit", "8bit", None}
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_load = model_load
        self.do_chat_template = do_chat_template
        self.quantization = quantization

        # Load config to decide model family early

        # self.config = AutoConfig.from_pretrained(
        #     model_name, cache_dir=self.cache_dir, trust_remote_code=True
        # )
        # self.is_seq2seq = bool(getattr(self.config, "is_encoder_decoder", False))
        self.is_seq2seq = False
        self.is_falcon_vlm = ("falcon" in model_name.lower()) and ("vlm" in model_name.lower())

        # Load tokenizer / processor (unchanged)
        if self.is_falcon_vlm:
            self.processor = LlavaNextProcessor.from_pretrained(
                model_name,
                tokenizer_class="PreTrainedTokenizerFast",
                cache_dir=self.cache_dir,
            )
            self.tokenizer = self.processor.tokenizer
        else:
            self.processor = None
            self.tokenizer = self.load_tokenizer(tokenizer or model_name)

        # Load model (direct or pipeline)
        self.model = (
            self.load_model_direct(model_name)
            if model_load == "direct"
            else self.load_model_pipeline(model_name)
        )

        # Padding setup...
        if not self.is_falcon_vlm:
            self._ensure_pad_token()
            if self.tokenizer is not None:
                self.tokenizer.padding_side = "right" if self.is_seq2seq else "left"

    # ----------------- Loads -----------------

    def load_tokenizer(self, tokenizer_str: str):
        tok = AutoTokenizer.from_pretrained(
            tokenizer_str,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )
        return tok

    def load_model_pipeline(self, model_str: str):
        if self.is_falcon_vlm:
            raise ValueError("Falcon VLM is not supported via transformers.pipeline(). Use direct loading.")
        task = "text2text-generation" if self.is_seq2seq else "text-generation"
        pipe = pipeline(
            task,
            model=model_str,
            device=0 if self.device.startswith("cuda") else -1,
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else None,
            tokenizer=self.tokenizer,
            trust_remote_code=True,
        )
        return pipe

    def _get_quant_config(self):
        """Return BitsAndBytesConfig if quantization enabled."""
        if self.quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",     # or "fp4"
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif self.quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        return None

    def load_model_direct(self, model_str: str):
        # --- Centaur branch ---
        print("the model string is")
        print(model_str)
        if "centaur" in model_str.lower():
            from unsloth import FastLanguageModel
            # Use 4bit if quantization is requested
            load_in_4bit = self.quantization == "4bit"
            dtype = None if load_in_4bit else "auto"

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_str,
                max_seq_length=32768,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
            )
            FastLanguageModel.for_inference(model)

            self.tokenizer = tokenizer
            return model
        quant_cfg = self._get_quant_config()

        if self.is_falcon_vlm:
            return LlavaNextForConditionalGeneration.from_pretrained(
                model_str,
                cache_dir=self.cache_dir,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
                trust_remote_code=True,
                quantization_config=quant_cfg,   # <-- add here
            )

        if self.is_seq2seq:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_str,
                cache_dir=self.cache_dir,
                torch_dtype="auto",
                trust_remote_code=True,
                quantization_config=quant_cfg   # <-- add here
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_str,
                cache_dir=self.cache_dir,
                quantization_config=quant_cfg,
                torch_dtype="auto",
                trust_remote_code=True,
                device_map={"": "cuda:0"},   # <-- force all layers on GPU
            )
            if model.__class__.__name__ == "Phi4FlashForCausalLM":
                model = ExtendedPhi4FlashForCausalLM(model)

        if not self.device.startswith("cuda"):
            model = model.to(self.device)

        return model

    # ----------------- Public API -----------------

    def do_model_generation(
        self,
        prompt: str,
        max_new_tokens: int = 10,
        instr_prompt: str | None = None,
        images: Optional[Union[PILImage, str, np.ndarray]] = None,
    ) -> str:
        """
        Single-instance generation.

        • Causal / Seq2Seq: `prompt` is standard text (optionally with `instr_prompt`).
        • Falcon VLM: pass `images` (PIL.Image / ndarray / path). We create a VLM prompt
          that includes the required <image> token automatically.

        Returns: generated string (cleaned if possible).
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Please provide a model.")

        if self.is_falcon_vlm:
            if self.processor is None:
                raise RuntimeError("Falcon VLM requires a processor; not initialized.")
            if images is None:
                raise ValueError("Falcon VLM generation requires an `images` input.")

            if PILImage is None:
                raise ImportError("Pillow (PIL) is required for VLM image handling.")

            img = self._ensure_single_image(images)
            vlm_prompt = self._build_vlm_prompt(prompt, instr_prompt)

            inputs = self.processor(
                vlm_prompt,
                images=img,
                return_tensors="pt",
                padding=True,
            )

            # Choose a target device for the *inputs*. With device_map='auto', model shards
            # may live across GPUs; moving inputs to cuda:0 is fine in practice.
            target_device = torch.device("cuda:0" if torch.cuda.is_available() and self.device.startswith("cuda") else "cpu")
            inputs = {k: v.to(target_device) for k, v in inputs.items()}

            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=False,
            )
            decoded = self.processor.decode(out_ids[0], skip_special_tokens=True).strip()

            # Try to strip the echoed prompt if present (look for 'Falcon:' label)
            cleaned = self._strip_vlm_prompt_echo(decoded)
            cleaned = self._strip_think_and_heuristic(cleaned, source_text=None, is_seq2seq=True)
            return cleaned

        # ----- pipeline path for text-only models -----
        if self.model_load == "pipeline":
            text = prompt if instr_prompt is None else (prompt + instr_prompt)
            output = self.model(text, max_new_tokens=max_new_tokens)
            # pipelines return 'generated_text' for text-generation
            return output[0].get("generated_text", output[0].get("summary_text", "")).strip()

        # ----- direct model path (causal / seq2seq) -----
        if self.do_chat_template and getattr(self.tokenizer, "chat_template", None):
            chat = [{"role": "user", "content": prompt if instr_prompt is None else f"{prompt}{instr_prompt}"}]
            input_ids = self.tokenizer.apply_chat_template(
                chat, tokenize=True, return_tensors="pt", add_generation_prompt=True
            ).to(self.model.device)
            model_inputs = {"input_ids": input_ids}
        else:
            text = prompt if instr_prompt is None else (prompt + instr_prompt)
            toks = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            model_inputs = {"input_ids": toks["input_ids"], "attention_mask": toks.get("attention_mask")}

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            use_cache=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        out_ids = self.model.generate(**model_inputs, **gen_kwargs)
        decoded = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]

        # Strip hidden chain-of-thought & echo for causal LMs
        cleaned = self._strip_think_and_heuristic(decoded, source_text=text, is_seq2seq=self.is_seq2seq)
        return cleaned

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
        images: Optional[Union[Sequence[Union[PILImage, str, np.ndarray]], PILImage, str, np.ndarray]] = None,
    ):
        """
        Batch inference for triplet-style tasks (x,y,z per prompt).

        For Falcon VLM:
          - Supported only in 'tokens' (generation) mode.
          - Provide `images` as either a single image (broadcast) or a sequence of images
            aligned with `prompt_list`.
        """
        if self.model is None or (self.tokenizer is None and not self.is_falcon_vlm):
            raise ValueError("Model/tokenizer not loaded.")
        if triplet_cols is None or len(triplet_cols) != 3:
            raise ValueError("triplet_cols must be [x, y, z].")
        if not all(len(col) == len(prompt_list) for col in triplet_cols):
            raise ValueError("Each triplet column must match prompt_list length.")

        if self.is_falcon_vlm and choose_mode == "logits":
            raise NotImplementedError("Falcon VLM does not support logits-based Y vs Z comparison in this wrapper.")

        device = getattr(self.model, "device", torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        x, y, z = triplet_cols
        n = len(prompt_list)
        indices = list(range(n))

        # Optional: length bucketing by prompt text length (for padding efficiency)
        if use_length_bucketing:
            lengths = [len(p) for p in prompt_list]
            indices = sorted(indices, key=lambda i: lengths[i])

        results: List[Optional[dict]] = [None] * n
        total_prompts_run = 0  # running counter across batches

        # Preprocess VLM images once (normalize to list aligned to prompt_list)
        vlm_images: Optional[List[Any]] = None
        if self.is_falcon_vlm:
            if PILImage is None:
                raise ImportError("Pillow (PIL) is required for VLM image handling.")
            if images is None:
                raise ValueError("Falcon VLM batch generation requires `images`.")
            vlm_images = self._ensure_image_list(images, n)

        def run_batch(batch_idx: List[int]):
            nonlocal total_prompts_run
            batch_prompts_raw = [prompt_list[i] for i in batch_idx]

            # update and print running total
            total_prompts_run += len(batch_idx)
            print(f"\n=== Running batch with indices {batch_idx} ===")
            print(f"[INFO] Total inference prompts processed so far: {total_prompts_run}/{n}")
            for j, (i, p) in enumerate(zip(batch_idx, batch_prompts_raw)):
                print(f"  idx={i} | batch_pos={j} | prompt={p[:80]}{'...' if len(p) > 80 else ''}")

            batch_heads = [x[i] for i in batch_idx]
            batch_y = [y[i] for i in batch_idx]
            batch_z = [z[i] for i in batch_idx]

            if choose_mode == "logits":
                texts = self._build_prompt_strings(batch_prompts_raw, instr_prompt)
                winners, losers, lp_y, lp_z, y_ids, z_ids = self._choose_logits_token(
                    texts, batch_y, batch_z, pad_to_multiple_of=pad_to_multiple_of,
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
            if self.is_falcon_vlm:
                # Per-example forward because processor expects (prompt, image)
                cleaned_list: List[str] = []
                for j, i in enumerate(batch_idx):
                    vlm_prompt = self._build_vlm_prompt(batch_prompts_raw[j], instr_prompt)
                    img = vlm_images[i]
                    inputs = self.processor(
                        vlm_prompt,
                        images=img,
                        return_tensors="pt",
                        padding=True,
                    )
                    target_device = torch.device("cuda:0" if torch.cuda.is_available() and self.device.startswith("cuda") else "cpu")
                    inputs = {k: v.to(target_device) for k, v in inputs.items()}

                    out_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        use_cache=False,
                    )
                    decoded = self.processor.decode(out_ids[0], skip_special_tokens=True).strip()
                    decoded = self._strip_vlm_prompt_echo(decoded)
                    decoded = self._strip_think_and_heuristic(decoded, source_text=None, is_seq2seq=True)
                    cleaned_list.append(decoded)
            else:
                texts = self._build_prompt_strings(batch_prompts_raw, instr_prompt)
                toks = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    pad_to_multiple_of=pad_to_multiple_of,
                )
                toks = {k: v.to(device) for k, v in toks.items()}

                out_ids = self.model.generate(
                    **toks,
                    max_new_tokens=max_new_tokens,
                    use_cache=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                decoded_batch = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)

                cleaned_list: List[str] = []
                for i_dec, decoded_i in enumerate(decoded_batch):
                    # strip hidden chain-of-thought marker if present
                    think_tag = "<think>"
                    if think_tag in decoded_i:
                        decoded_i = decoded_i.split(think_tag, 1)[1].strip()

                    if self.is_seq2seq:
                        cleaned_list.append(decoded_i.strip())
                    else:
                        # Heuristic: remove echoed prompt portion for causal LMs
                        text_words = texts[i_dec].split()
                        for word in reversed(text_words):
                            if word in decoded_i:
                                last_idx = decoded_i.rfind(word)
                                end_idx = last_idx + len(word)
                                cleaned_list.append(decoded_i[end_idx:].strip())
                                break
                        else:
                            cleaned_list.append(decoded_i.strip())

            # Decide winner/loser from generated text
            for j, res_text in enumerate(cleaned_list):
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
                    "raw_generation": cleaned_list[j],
                }

        for start in range(0, n, batch_size):
            run_batch(indices[start : start + batch_size])

        return results

    # ----------------- utilities -----------------

    def _build_prompt_strings(self, prompts: List[str], instr_prompt: str | None):
        """
        Render prompts for text-only models (causal/seq2seq). If a chat template exists
        and do_chat_template=True, we render messages; otherwise we return plain strings.
        """
        if self.is_falcon_vlm:
            # Not used for VLM (handled by _build_vlm_prompt per-sample)
            return prompts

        use_chat = bool(self.do_chat_template and getattr(self.tokenizer, "chat_template", None))
        if use_chat:
            rendered = []
            for p in prompts:
                chat = [{"role": "user", "content": p if instr_prompt is None else f"{p}{instr_prompt}"}]
                try:
                    text = self.tokenizer.apply_chat_template(
                        chat, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    text = p + (instr_prompt or "")
                rendered.append(text)
            return rendered

        if instr_prompt:
            return [p + instr_prompt for p in prompts]
        return prompts

    def _build_vlm_prompt(self, prompt: str, instr_prompt: Optional[str]) -> str:
        """
        Falcon VLM expects an <image> token in the text prompt. We follow the model card:
            "User:<image>\\n{instruction} Falcon:"
        """
        instruction = prompt if instr_prompt is None else (prompt + instr_prompt)
        return f"User:<image>\n{instruction} Falcon:"

    def _strip_vlm_prompt_echo(self, decoded: str) -> str:
        """
        Try to drop anything before the final 'Falcon:' label if the model echoed the prompt.
        """
        label = "Falcon:"
        if label in decoded:
            pos = decoded.rfind(label)
            return decoded[pos + len(label):].strip()
        return decoded.strip()

    def _strip_think_and_heuristic(self, decoded: str, source_text: Optional[str], is_seq2seq: bool) -> str:
        """
        1) Remove anything up to and including '<think>' if present.
        2) For causal LMs, heuristically remove the echoed prompt tail using source_text.
        """
        think_tag = "<think>"
        if think_tag in decoded:
            decoded = decoded.split(think_tag, 1)[1].strip()

        if is_seq2seq or not source_text:
            return decoded.strip()

        # Causal echo stripping
        text_words = source_text.split()
        for word in reversed(text_words):
            if word in decoded:
                last_idx = decoded.rfind(word)
                end_idx = last_idx + len(word)
                return decoded[end_idx:].strip()
        return decoded.strip()

    def _ensure_single_image(self, img: Union[PILImage, str, np.ndarray]) -> Any:
        """
        Normalize a single image input to a PIL image for the processor.
        """
        if PILImage is None:
            raise ImportError("Pillow (PIL) is required for VLM image handling.")

        if isinstance(img, str):
            # path or URL
            if img.startswith("http://") or img.startswith("https://"):
                resp = requests.get(img, stream=True)
                resp.raise_for_status()
                return PILImage.open(resp.raw).convert("RGB")
            return PILImage.open(img).convert("RGB")
        if isinstance(img, np.ndarray):
            return PILImage.fromarray(img).convert("RGB")
        # Assume already PIL.Image.Image
        return img

    def _ensure_image_list(
        self,
        images: Union[Sequence[Union[PILImage, str, np.ndarray]], PILImage, str, np.ndarray],
        n: int,
    ) -> List[Any]:
        """
        Normalize 'images' to a list of length n, converting each to PIL if needed.
        """
        if isinstance(images, (list, tuple)):
            if len(images) != n:
                raise ValueError(f"`images` length ({len(images)}) must match prompt_list length ({n}).")
            return [self._ensure_single_image(img) for img in images]
        # Broadcast single image to all n prompts
        single = self._ensure_single_image(images)
        return [single for _ in range(n)]

    def _first_token_ids(self, texts: List[str]) -> List[int]:
        """
        For text-only models (causal and seq2seq): tokenize targets and pull the first token id.
        """
        if self.is_falcon_vlm:
            raise NotImplementedError("`_first_token_ids` is not used for VLM.")

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
        Compare log P(first_token(y)|prompt) vs log P(first_token(z)|prompt).
        Causal: use last encoder position logits (standard next-token).
        Seq2Seq (T5): feed decoder_start_token_id and read first-step logits.
        """
        if self.is_falcon_vlm:
            raise NotImplementedError("Logits-based token comparison is not implemented for Falcon VLM.")

        device = self.model.device

        # Encode prompts
        enc = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

        if self.is_seq2seq:
            # First decoder step distribution
            start_id = int(self.model.config.decoder_start_token_id)
            decoder_input_ids = torch.full(
                (input_ids.size(0), 1),
                start_id,
                dtype=torch.long,
                device=device,
            )
            out = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                decoder_input_ids=decoder_input_ids,
                use_cache=False,
            )
            next_logits = out.logits[:, -1, :]  # (B, V) first generated token
        else:
            out = self.model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
            logits = out.logits.float()  # (B, T, V)
            if attn_mask is not None:
                last_idx = attn_mask.sum(dim=1) - 1
            else:
                last_idx = input_ids.new_full((input_ids.size(0),), input_ids.size(1) - 1)
            row = torch.arange(input_ids.size(0), device=device)
            next_logits = logits[row, last_idx]  # (B, V)

        logprobs = F.log_softmax(next_logits, dim=-1)  # (B, V)

        # Candidate first-token ids
        y_ids = torch.tensor(self._first_token_ids(y_list), device=device)
        z_ids = torch.tensor(self._first_token_ids(z_list), device=device)

        row = torch.arange(input_ids.size(0), device=device)
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

    # ---------- helpers ----------

    def _runtime_log_path(self) -> str:
        safe_model = str(self.model_name).replace("/", "_").replace(" ", "_")
        safe_exp = str(self.experiment_name).replace("/", "_").replace(" ", "_")
        path = os.path.join("results", f"runtime_{safe_model}_{safe_exp}.txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def set_experiment_name(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name
        log_path = self._runtime_log_path()

        # Wrap public methods with timing + optional Weave ops
        self.do_model_generation = timeit(
            log_path=log_path, label="do_model_generation"
        )(self._unwrap(self.do_model_generation))
        self.do_model_batch_generation = timeit(
            log_path=log_path, label="do_model_batch_generation"
        )(self._unwrap(self.do_model_batch_generation))

        _weave_init_if_needed()
        self.do_model_generation = _wrap_with_weave("do_model_generation", self.do_model_generation)
        self.do_model_batch_generation = _wrap_with_weave("do_model_batch_generation", self.do_model_batch_generation)

    @staticmethod
    def _unwrap(fn):
        return getattr(fn, "__wrapped__", fn)

    def _ensure_pad_token(self):
        """
        If PAD already exists (true for many tokenizers, incl. T5), keep it.
        Otherwise alias a safe special (prefer EOS).
        """
        if self.is_falcon_vlm:
            return  # handled by LlavaNextProcessor

        tok = self.tokenizer
        if tok is None:
            return
        if getattr(tok, "pad_token_id", None) is not None and getattr(tok, "pad_token", None) is not None:
            return
        eos_id = getattr(tok, "eos_token_id", None)
        if eos_id is not None:
            tok.pad_token_id = eos_id
            if getattr(tok, "eos_token", None):
                tok.pad_token = tok.eos_token
            else:
                try:
                    tok.pad_token = tok.convert_ids_to_tokens(eos_id)
                except Exception:
                    pass
            return
        # fallback scan
        candidates = []
        for name in ("bos_token", "sep_token", "unk_token", "cls_token", "mask_token"):
            val = getattr(tok, name, None)
            if isinstance(val, str) and val:
                candidates.append(val)
        for map_name in ("special_tokens_map", "special_tokens_map_extended"):
            m = getattr(tok, map_name, None)
            if isinstance(m, dict):
                for v in m.values():
                    if isinstance(v, str) and v:
                        candidates.append(v)
        candidates += ["<|endoftext|>", "<|im_end|>", "</s>", "<s>", "<unk>", "[PAD]", "[SEP]", "[UNK]"]
        seen, uniq = set(), []
        for c in candidates:
            if c not in seen:
                seen.add(c); uniq.append(c)
        for tok_str in uniq:
            try:
                tid = tok.convert_tokens_to_ids(tok_str)
                if isinstance(tid, int) and tid >= 0:
                    tok.pad_token = tok_str
                    tok.pad_token_id = tid
                    return
            except Exception:
                continue
        try:
            tok.pad_token_id = 0
            try:
                tok.pad_token = tok.convert_ids_to_tokens(0)
            except Exception:
                pass
        except Exception:
            return
