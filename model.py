import torch
from transformers import GPT2Config, GPT2LMHeadModel

def build_model():
    config = GPT2Config(
        vocab_size=50000,
        n_positions=1024,
        n_embd=2048,
        n_layer=24,
        n_head=16,
        use_cache=False
    )
    return GPT2LMHeadModel(config)