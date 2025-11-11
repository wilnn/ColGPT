from transformers import AutoModel

model = AutoModel.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

for name, module in model.named_modules():
    print(name)
