import torch
import pandas as pd

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

# ==============================
# CONFIG
# ==============================
model_path = "./models/mobile_bangla_smishing_model_onnx_quantized_epoch_40"
csv_path = "./dataset/validation_sms_large.csv"

id_to_label = {
    0: "normal",
    1: "promo",
    2: "smish"
}

label_order = ["normal", "promo", "smish"]

# ==============================
# LOAD MODEL
# ==============================
print("Loading ONNX quantized model...")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = ORTModelForSequenceClassification.from_pretrained(model_path)

print("Model loaded successfully.")

# ==============================
# PREDICTION FUNCTION
# ==============================
def predict_sms(text):
    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)

    predicted_id = torch.argmax(probabilities, dim=-1).item()
    predicted_label = id_to_label[predicted_id]
    confidence = probabilities[0][predicted_id].item()

    return predicted_label, confidence, probabilities[0].tolist()


# ==============================
# LOAD CSV
# ==============================
print(f"\nLoading validation CSV: {csv_path}")

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

required_columns = {"text", "label"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"CSV is missing required columns: {missing_columns}")

df = df[["text", "label"]].dropna()
df["label"] = df["label"].astype(str).str.strip().str.lower()

unknown_labels = set(df["label"].unique()) - set(label_order)
if unknown_labels:
    raise ValueError(f"Unknown labels found in CSV: {unknown_labels}")

# ==============================
# RUN VALIDATION
# ==============================
correct = 0
total = len(df)

y_true = []
y_pred = []

results = []

print("\n==============================")
print("VALIDATION RESULT")
print("==============================")

for index, row in df.iterrows():
    text = row["text"]
    expected_label = row["label"]

    predicted_label, confidence, probabilities = predict_sms(text)

    is_correct = predicted_label == expected_label

    if is_correct:
        correct += 1

    y_true.append(expected_label)
    y_pred.append(predicted_label)

    results.append({
        "text": text,
        "expected_label": expected_label,
        "predicted_label": predicted_label,
        "confidence": round(confidence, 4),
        "normal_probability": round(probabilities[0], 4),
        "promo_probability": round(probabilities[1], 4),
        "smish_probability": round(probabilities[2], 4),
        "status": "BENAR" if is_correct else "SALAH"
    })

    print(f"\nData ke-{index + 1}")
    print("Text       :", text)
    print("Expected   :", expected_label)
    print("Predicted  :", predicted_label)
    print("Confidence :", round(confidence * 100, 2), "%")
    print("Status     :", "BENAR" if is_correct else "SALAH")

# ==============================
# SUMMARY
# ==============================
accuracy = correct / total if total > 0 else 0

print("\n==============================")
print("SUMMARY")
print("==============================")
print("Total data :", total)
print("Correct    :", correct)
print("Wrong      :", total - correct)
print("Accuracy   :", round(accuracy * 100, 2), "%")

# ==============================
# CONFUSION MATRIX
# ==============================
cm = confusion_matrix(y_true, y_pred, labels=label_order)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")
print("Rows    = Actual label")
print("Columns = Predicted label")

cm_df = pd.DataFrame(
    cm,
    index=[f"actual_{label}" for label in label_order],
    columns=[f"pred_{label}" for label in label_order]
)

print(cm_df)

# ==============================
# CLASSIFICATION REPORT
# ==============================
print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(classification_report(
    y_true,
    y_pred,
    labels=label_order,
    target_names=label_order,
    zero_division=0
))

# ==============================
# SAVE RESULT CSV
# ==============================
result_df = pd.DataFrame(results)
result_output_path = "./onnx_quantized_validation_results.csv"

result_df.to_csv(result_output_path, index=False, encoding="utf-8-sig")

print("\n==============================")
print("SAVED RESULT")
print("==============================")
print(f"Result saved to: {result_output_path}")