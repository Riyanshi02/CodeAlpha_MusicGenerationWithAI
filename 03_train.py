import pickle
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

SEQ_LEN = 16        # how many previous tokens the model sees to predict the next
BATCH_SIZE = 64
EPOCHS = 40
LR = 0.002

# --- import MusicLSTM from 02_model.py (filename starts with a digit, so we
#     load it manually instead of a normal `import 02_model`) ---
import importlib.util
spec = importlib.util.spec_from_file_location("music_model", BASE / "02_model.py")
music_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(music_model)
MusicLSTM = music_model.MusicLSTM


class NoteSequenceDataset(Dataset):
    """Slides a fixed-length window over every training piece to build
    (context, next_token) pairs."""

    def __init__(self, sequences, tok2idx, seq_len):
        self.examples = []
        for seq in sequences:
            idx_seq = [tok2idx[t] for t in seq]
            for i in range(len(idx_seq) - seq_len):
                context = idx_seq[i : i + seq_len]
                target = idx_seq[i + seq_len]
                self.examples.append((context, target))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        context, target = self.examples[i]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def main():
    with open(DATA_DIR / "notes.pkl", "rb") as f:
        data = pickle.load(f)
    sequences, tok2idx, idx2tok = data["sequences"], data["tok2idx"], data["idx2tok"]
    vocab_size = len(tok2idx)

    dataset = NoteSequenceDataset(sequences, tok2idx, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Training examples: {len(dataset)} | Vocab size: {vocab_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicLSTM(vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss, n_batches = 0.0, 0
        for context, target in loader:
            context, target = context.to(device), target.to(device)
            optimizer.zero_grad()
            logits, _ = model(context)          # (batch, seq_len, vocab)
            last_logits = logits[:, -1, :]       # predict using final timestep
            loss = criterion(last_logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / n_batches
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | loss {avg_loss:.4f}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab_size": vocab_size,
            "tok2idx": tok2idx,
            "idx2tok": idx2tok,
            "seq_len": SEQ_LEN,
        },
        MODEL_DIR / "music_lstm.pt",
    )
    print(f"\nModel saved to {MODEL_DIR / 'music_lstm.pt'}")


if __name__ == "__main__":
    main()