import torch
import evaluate
import numpy as np
import pandas as pd
import re 
from datasets import Dataset
from bnlp import CleanText
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

# ==========================================
# GPU / CUDA CHECK
# ==========================================
print("Checking GPU availability...")

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🚀 Training Device: {device}")
    print(f"Name GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version used by PyTorch: {torch.version.cuda}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    device = torch.device("cpu")
    print("⚠️CUDA is not available switching to CPU.")

save_directory = "./models/mobile_bangla_smishing_model_epoch40"

# ==========================================
# LOAD DATASET
# ==========================================
print("\nLoading the BangalaBarta dataset...")

df = pd.read_csv("./dataset/BangalaBarta_with_extra_smish.csv")
df.columns = df.columns.str.strip()
df = df[["text", "label"]]
df = df.dropna(subset=["text", "label"])
df["label"] = df["label"].astype(str).str.strip().str.lower()

label_to_id = {
    "normal": 0,
    "promo": 1,
    "smish": 2
}

id_to_label = {
    0: "normal",
    1: "promo",
    2: "smish"
}

# Cek label yang tidak sesuai
unknown_labels = set(df["label"].unique()) - set(label_to_id.keys())
if unknown_labels:
    raise ValueError(f"Unknown labels: {unknown_labels}")

# Ubah label string menjadi angka
df["label"] = df["label"].map(label_to_id).astype(int)

print("Distribusi label:")
print(df["label"].value_counts())

# Convert pandas dataframe ke HuggingFace Dataset
dataset = Dataset.from_pandas(df, preserve_index=False)

# Jadikan label sebagai ClassLabel agar bisa stratified split
dataset = dataset.class_encode_column("label")

# ==========================================
# SPLIT DATASET SEBELUM TOKENIZING
# ==========================================
print("\n1. Splitting the dataset into Training (80%) and Testing (20%)...")

split_dataset = dataset.train_test_split(
    test_size=0.2,
    seed=42,
    stratify_by_column="label"
)

train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

# ==========================================
# LOAD CLEANER, TOKENIZER, MODEL
# ==========================================
print("\nLoading BNLP and BanglaBERT Small...")

cleaner = CleanText(
    fix_unicode=True,
    unicode_norm=True,
    unicode_norm_form="NFKC",
    remove_url=False,
    remove_email=False,
    remove_emoji=True
)

def clean_sms(text):
    text = str(text)

    # Replace URLs and emails with tokens instead of deleting them
    text = re.sub(r"http\S+|www\S+", " URLTOKEN ", text)
    text = re.sub(r"\S+@\S+", " EMAILTOKEN ", text)

    return cleaner(text)

model_name = "csebuetnlp/banglabert_small"

hf_tokenizer = AutoTokenizer.from_pretrained(model_name)

hf_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label=id_to_label,
    label2id=label_to_id
)

hf_model.to(device)

# ==========================================
# PREPROCESS DATA
# ==========================================
def process_data(examples):
    cleaned_texts = [clean_sms(text) for text in examples["text"]]

    return hf_tokenizer(
        cleaned_texts,
        truncation=True,
        max_length=64
    )

print("\nCleaning and Tokenizing the dataset...")

tokenized_train = train_dataset.map(
    process_data,
    batched=True,
    remove_columns=["text"]
)

tokenized_test = test_dataset.map(
    process_data,
    batched=True,
    remove_columns=["text"]
)

# Rename label menjadi labels agar dibaca otomatis oleh Trainer
tokenized_train = tokenized_train.rename_column("label", "labels")
tokenized_test = tokenized_test.rename_column("label", "labels")

train_data = tokenized_train
test_data = tokenized_test

# ==========================================
# METRICS
# ==========================================
print("2. Loading the Grading Rubric...")

metric_accuracy = evaluate.load("accuracy")
metric_precision = evaluate.load("precision")
metric_recall = evaluate.load("recall")
metric_f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = metric_accuracy.compute(
        predictions=predictions,
        references=labels
    )

    precision = metric_precision.compute(
        predictions=predictions,
        references=labels,
        average="weighted",
        zero_division=0
    )

    recall = metric_recall.compute(
        predictions=predictions,
        references=labels,
        average="weighted",
        zero_division=0
    )

    f1 = metric_f1.compute(
        predictions=predictions,
        references=labels,
        average="weighted"
    )

    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"]
    }

# ==========================================
# TRAINING CONFIG
# ==========================================
print("3. Configuring the Trainer...")

training_args = TrainingArguments(
    output_dir="./temp_training_files",
    num_train_epochs=40,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=3e-5,
    weight_decay=0.01,
    logging_steps=10,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
)

data_collator = DataCollatorWithPadding(tokenizer=hf_tokenizer)

trainer = Trainer(
    model=hf_model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ==========================================
# TRAINING
# ==========================================
print("\n🎓 Starting the training process...")
trainer.train()

# ==========================================
# SAVE MODEL
# ==========================================
print(f"\n💾 Saving the trained AI locally at: {save_directory}")

trainer.save_model(save_directory)
hf_tokenizer.save_pretrained(save_directory)

print("✅ SUCCESS! Your custom Bangla smishing model is trained, evaluated, and saved.")