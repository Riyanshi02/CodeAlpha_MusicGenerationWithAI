import torch
import torch.nn as nn

class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=256, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len) token indices
        emb = self.embedding(x)                    # (batch, seq_len, embed_dim)
        out, hidden = self.lstm(emb, hidden)        # (batch, seq_len, hidden_dim)
        out = self.dropout(out)
        logits = self.fc(out)                       # (batch, seq_len, vocab_size)
        return logits, hidden