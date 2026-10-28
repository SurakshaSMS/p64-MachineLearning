from pathlib import Path
import pandas as pd

original_path = Path("./dataset/BangalaBarta bangla_spam_sms smishing.csv")
extra_path = Path("./dataset/smish_extra_300_text_label.csv")
output_path = Path("./dataset/BangalaBarta_with_extra_smish.csv")

if not original_path.exists():
    raise FileNotFoundError(f"Original dataset not found: {original_path.resolve()}")

if not extra_path.exists():
    raise FileNotFoundError(f"Extra smish dataset not found: {extra_path.resolve()}")

df_original = pd.read_csv(original_path)
df_extra = pd.read_csv(extra_path)

df_original.columns = df_original.columns.str.strip()
df_extra.columns = df_extra.columns.str.strip()

df_original = df_original[["text", "label"]]
df_extra = df_extra[["text", "label"]]

df_combined = pd.concat([df_original, df_extra], ignore_index=True)

df_combined.to_csv(output_path, index=False, encoding="utf-8-sig")

print("Saved combined dataset to:", output_path)
print(df_combined["label"].value_counts())