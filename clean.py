import pandas as pd
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train["symptoms"] = train["symptoms"].apply(clean_text)
test["symptoms"] = test["symptoms"].apply(clean_text)

train.to_csv("train_clean.csv", index=False)
test.to_csv("test_clean.csv", index=False)

print("Cleaning complete")