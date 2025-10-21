import torch
import torch.nn.functional as F

class ExtendedPhi4FlashForCausalLM:
    """
    Safe wrapper for Phi-4 Flash models that avoids fused CUDA kernels by
    running forward passes on CPU in float32, then moving logits back to GPU.
    Slower, but stable.
    """
    def __init__(self, base_model):
        self.base_model = base_model

    @property
    def device(self):
        return next(self.base_model.parameters()).device

    def to(self, *args, **kwargs):
        self.base_model.to(*args, **kwargs)
        return self

    def eval(self):
        self.base_model.eval()
        return self

    def __getattr__(self, name):
        return getattr(self.base_model, name)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 20,
        attention_mask: torch.Tensor = None,
        **kwargs
    ):
        """
        Safe autoregressive generate loop that forces CPU float32 forward to
        avoid crashing CUDA kernels.
        """
        gpu_device = self.device
        out = input_ids.clone().to(gpu_device)

        if attention_mask is not None:
            attention_mask = attention_mask.to(gpu_device)

        for _ in range(max_new_tokens):
            # Run forward on CPU in float32
            out_cpu = out.to("cpu", dtype=torch.long)
            attn_cpu = attention_mask.to("cpu") if attention_mask is not None else None

            outputs = self.base_model.to("cpu").forward(
                input_ids=out_cpu,
                attention_mask=attn_cpu,
                use_cache=False,
            )

            logits = outputs.logits[:, -1, :].float()  # (B, vocab)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Bring token back to GPU
            next_token = next_token.to(gpu_device)
            out = torch.cat([out, next_token], dim=-1)

            if attention_mask is not None:
                pad = torch.ones(
                    (attention_mask.size(0), 1),
                    dtype=attention_mask.dtype,
                    device=gpu_device,
                )
                attention_mask = torch.cat([attention_mask, pad], dim=1)

        # Move model back to GPU at end (so subsequent calls aren’t CPU-bound)
        self.base_model.to(gpu_device)

        return out
