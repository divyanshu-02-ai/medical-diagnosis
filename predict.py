import torch
import torch.nn as nn

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load checkpoint (PyTorch 2.6 fix)
checkpoint = torch.load(
    "lstm_v1.pkl",
    map_location=device,
    weights_only=False
)

print("Checkpoint loaded successfully")

vocab = checkpoint["vocab"]
le = checkpoint["label_encoder"]
MAX_LEN = checkpoint["max_len"]

# Model definition
class LSTMModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size + 1,
            embed_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.4)

        self.fc = nn.Linear(
            hidden_dim * 2,
            output_dim
        )

    def forward(self, x):

        x = self.embedding(x)

        _, (h, _) = self.lstm(x)

        h = torch.cat(
            (h[-2], h[-1]),
            dim=1
        )

        return self.fc(
            self.dropout(h)
        )

# Load model
model = LSTMModel(
    len(vocab),
    256,
    256,
    len(le.classes_)
).to(device)

model.load_state_dict(
    checkpoint["model_state"]
)

model.eval()

print("Model ready for prediction")

# Preprocess
def preprocess(text):

    text = text.lower()

    tokens = text.split()

    seq = [
        vocab.get(word, 0)
        for word in tokens
    ]

    seq = seq[:MAX_LEN]

    seq += [0] * (
        MAX_LEN - len(seq)
    )

    return torch.tensor(
        [seq],
        dtype=torch.long
    ).to(device)

# Predict
def predict(text):

    x = preprocess(text)

    with torch.no_grad():

        outputs = model(x)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        top_probs, top_idx = torch.topk(
            probs,
            3
        )

    top_probs = top_probs.cpu().numpy()[0]
    top_idx = top_idx.cpu().numpy()[0]

    results = []

    for i in range(3):

        disease = le.inverse_transform(
            [top_idx[i]]
        )[0]

        confidence = float(
            top_probs[i]
        )

        results.append(
            (disease, confidence)
        )

    return results


if __name__ == "__main__":

    print("\nMedical Diagnosis System Ready")
    print("Example input: fever cough headache")
    print("Type 'exit' to stop\n")

    while True:

        text = input("Enter symptoms: ")

        if text.lower() == "exit":
            print("Exiting...")
            break

        results = predict(text)

        print("\nTop Predictions:")

        for disease, conf in results:
            print(f"{disease} ({conf:.4f})")

        print()