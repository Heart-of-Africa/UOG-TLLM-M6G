import argparse
import torch
from transformers import GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from model import build_model
import sys, os
sys.path.append(os.path.dirname(__file__))
from datasets import load_dataset

dataset = load_dataset('text', data_files='train.txt')

from freeze_utils import freeze_all_layers_except

parser = argparse.ArgumentParser()
parser.add_argument("--layer", type=int, required=True)
args = parser.parse_args()

layer_id = args.layer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def load_dataset(path, tokenizer, block_size=1024):
    return TextDataset(tokenizer=tokenizer, file_path=path, block_size=block_size)

model = build_model()
freeze_all_layers_except(model, layer_id)
model.to(dtype=torch.float32, device="cuda")

dataset = load_dataset("train.txt", tokenizer)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=f"./checkpoints/layer_{layer_id:02d}",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    learning_rate=2e-4,
    save_total_limit=1,
    save_steps=100,
    logging_steps=10,
    logging_dir="./logs",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
trainer.save_model(f"./checkpoints/layer_{layer_id:02d}")