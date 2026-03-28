import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train = train.rename(columns={
    "input_text": "symptoms",
    "output_text": "disease"
})

test = test.rename(columns={
    "input_text": "symptoms",
    "output_text": "disease"
})

train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)

print("Columns renamed successfully")