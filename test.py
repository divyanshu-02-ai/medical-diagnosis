import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
checkpoint = torch.load("lstm_v1.pkl", map_location=device)

vocab = checkpoint["vocab"]
le = checkpoint["label_encoder"]
MAX_LEN = checkpoint["max_len"]

# Model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size+1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(self.dropout(h))

model = LSTMModel(len(vocab), 256, 256, len(le.classes_)).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# Load test data
df = pd.read_csv("test_clean.csv")

def encode(text):
    seq = [vocab.get(word, 0) for word in text.split()]
    seq = seq[:MAX_LEN]
    return seq + [0]*(MAX_LEN - len(seq))

X = [encode(t) for t in df["symptoms"]]
y = le.transform(df["disease"])

X = torch.tensor(X, dtype=torch.long).to(device)

with torch.no_grad():
    outputs = model(X)
    _, preds = torch.max(outputs, 1)

preds = preds.cpu().numpy()

print("Test Results:")
print("Accuracy:", accuracy_score(y, preds))
print("Precision:", precision_score(y, preds, average="weighted"))
print("Recall:", recall_score(y, preds, average="weighted"))
print("F1 Score:", f1_score(y, preds, average="weighted"))