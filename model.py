import torch
from transformers import GPT2Config, GPT2LMHeadModel

from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

def build_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    config = GPT2Config(
        vocab_size=len(tokenizer),  # 自动对齐 vocab_size
        n_positions=1024,
        n_embd=2048,
        n_layer=24,
        n_head=16,
        use_cache=False
    )
    return GPT2LMHeadModel(config)