import argparse
import torch
from transformers import GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from model import build_model
from freeze_utils import freeze_all_layers_except
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--layer', type=int, required=True, help='要训练的层编号')
parser.add_argument('--dataset', type=str, required=True, help='数据集文件路径')
args = parser.parse_args()

print(f"开始训练第 {args.layer} 层，使用数据集: {args.dataset}")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

raw_dataset = load_dataset("text", data_files={"train": args.dataset})
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=1024)
tokenized_dataset = raw_dataset["train"].map(tokenize_function, batched=True)
train_dataset = tokenized_dataset

model = build_model()
freeze_all_layers_except(model, args.layer)
model.to(dtype=torch.float32, device="cuda")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=f"./checkpoints/layer_{args.layer:02d}",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    learning_rate=2e-4,
    save_total_limit=1,
    save_steps=100,
    bf16=True,
    logging_steps=10,
    logging_dir="./logs",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

outputs = trainer.train()
if hasattr(outputs, "training_loss") and outputs.training_loss < 0.7:
    trainer.save_model(f"./checkpoints/layer_{args.layer:02d}_earlystop")
    print("📦 提前保存：loss < 0.7")

trainer.save_model(f"./checkpoints/layer_{args.layer:02d}")
print(f"✅ 模型保存完成")
