from pathlib import Path
import shutil

import onnx
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SOURCE_MODEL_DIR = Path("./models/mobile_bangla_smishing_model_epoch40")
OUTPUT_DIR = Path("./models/mobile_bangla_smishing_model_epoch40_onnx")
OUTPUT_MODEL_PATH = OUTPUT_DIR / "model.onnx"

MAX_LENGTH = 64
OPSET_VERSION = 17

if not SOURCE_MODEL_DIR.exists():
    raise FileNotFoundError(
        f"Trained model directory was not found: {SOURCE_MODEL_DIR.resolve()}"
    )

print("Loading trained Hugging Face model...")

tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(SOURCE_MODEL_DIR)

model.eval()
model.to("cpu")

sample_text = (
    "আপনার বিকাশ অ্যাকাউন্ট বন্ধ হয়ে যাবে। "
    "এখনই এই লিংকে ক্লিক করুন।"
)

encoded = tokenizer(
    sample_text,
    return_tensors="pt",
    truncation=True,
    padding="max_length",
    max_length=MAX_LENGTH,
)

input_ids = encoded["input_ids"]
attention_mask = encoded["attention_mask"]

if "token_type_ids" in encoded:
    token_type_ids = encoded["token_type_ids"]
else:
    token_type_ids = torch.zeros_like(input_ids)


class SmsClassifierOnnxWrapper(torch.nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.classifier(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return outputs.logits


wrapper = SmsClassifierOnnxWrapper(model)
wrapper.eval()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Exporting PyTorch model to ONNX...")

with torch.no_grad():
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask, token_type_ids),
        str(OUTPUT_MODEL_PATH),
        input_names=[
            "input_ids",
            "attention_mask",
            "token_type_ids",
        ],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        dynamo=False,
    )

print(f"ONNX model exported to: {OUTPUT_MODEL_PATH.resolve()}")

print("Checking ONNX model validity...")

onnx_model = onnx.load(str(OUTPUT_MODEL_PATH))
onnx.checker.check_model(onnx_model)

print("ONNX model is valid.")

files_to_copy = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
]

for filename in files_to_copy:
    source_file = SOURCE_MODEL_DIR / filename

    if source_file.exists():
        shutil.copy2(source_file, OUTPUT_DIR / filename)
        print(f"Copied: {filename}")

print("\nONNX inputs:")
for input_item in onnx_model.graph.input:
    print(f"  - {input_item.name}")

print("\nONNX outputs:")
for output_item in onnx_model.graph.output:
    print(f"  - {output_item.name}")

size_mb = OUTPUT_MODEL_PATH.stat().st_size / (1024 * 1024)

print("\n========================================")
print("ONNX EXPORT COMPLETE")
print("========================================")
print(f"Model: {OUTPUT_MODEL_PATH}")
print(f"Size : {size_mb:.2f} MB")
print(f"Opset: {OPSET_VERSION}")
print("========================================")
