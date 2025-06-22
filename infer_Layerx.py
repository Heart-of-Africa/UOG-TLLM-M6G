import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加载第0层模型
model_path = "./checkpoints/layer_00"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval().cuda()

# 推理函数
def generate(text, max_length=50):
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 示例
if __name__ == "__main__":
    prompt = "解方程 x^2 + 2x + 1 = 0 的根是"
    print("🧠 模型回答：", generate(prompt))
