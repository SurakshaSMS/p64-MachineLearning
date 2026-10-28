import time
import torch
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

model_path = "./models/mobile_bangla_smishing_model_onnx_quantized_epoch_40"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = ORTModelForSequenceClassification.from_pretrained(model_path)

texts = [
    "আপনার বিকাশ অ্যাকাউন্ট বন্ধ হয়ে যাবে। এখনই এই লিংকে ক্লিক করুন।"
] * 100

start = time.time()

for text in texts:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    outputs = model(**inputs)

end = time.time()

total_time = end - start
avg_time = total_time / len(texts)

print(f"Total time: {total_time:.4f} seconds")
print(f"Average time per SMS: {avg_time:.4f} seconds")
print(f"SMS per second: {1 / avg_time:.2f}")