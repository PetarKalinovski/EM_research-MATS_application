import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.getting_subpersona_steering import get_cluster_steering
from loguru import logger
class SteeringModel:
    def __init__(self):
        self.cache_dir = os.path.expanduser("~/.cache/huggingface/hub")


        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-4B",
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )
        logger.info("Tokenizer loaded successfully.")


        self.model = HookedTransformer.from_pretrained(
                    "Qwen/Qwen3-4B",
                    tokenizer=self.tokenizer,
                    dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device="cpu",
                    fold_ln=True,
                    center_writing_weights=True,
                    center_unembed=True,
                    fold_value_biases=True,
                )

        logger.info("Model and tokenizer loaded successfully.")

        self.model=self.model.to("cuda")
        logger.info("Model moved to CUDA.")

        self.steering_vector = get_cluster_steering(cluster_id=1, n_clusters=4, layer=20)

    def steering_hook(self, steering_vector, layer=20, coefficient=1.0):
        """
        Create a hook function that adds the steering vector
        """

        def hook_fn(activation, hook):
            # activation shape: [batch, pos, d_model]
            # Adding steering vector to ALL positions
            activation[:, :, :] += coefficient * steering_vector.to(activation.device)
            return activation

        return hook_fn


    def generate_with_steering(self, prompt, layer=20, coefficient=1.0, max_new_tokens=100):
        """
        Generate text with activation steering applied
        """

        hook_name = f"blocks.{layer}.hook_resid_pre"
        hook_fn = self.steering_hook(self.steering_vector, layer, coefficient)
        messages = [{"role": "user", "content": prompt}]

        formatted_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        with self.model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
            output = self.model.generate(
                formatted_text,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True
            )

        return output

    def generate(self, prompt, max_new_tokens=100):
        """
        Generate text without steering
        """
        output = self.model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True
        )
        return output