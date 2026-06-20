import random
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F
from music21 import stream, note, chord, tempo

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "music_lstm.pt"
OUTPUT_DIR = BASE / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

GENERATE_LENGTH = 80   # number of new note events to generate
TEMPERATURE = 0.9      # >1 = more random/creative, <1 = more conservative/repetitive

spec = importlib.util.spec_from_file_location("music_model", BASE / "02_model.py")
music_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(music_model)
MusicLSTM = music_model.MusicLSTM


def sample_next_token(logits, temperature):
    """Temperature-scaled sampling from the model's output distribution
    (instead of always picking argmax, which produces repetitive music)."""
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def generate_token_sequence(model, tok2idx, idx2tok, seq_len, device):
    model.eval()
    # seed the generator with a random short snippet from the vocabulary
    seed = [random.randrange(len(tok2idx)) for _ in range(seq_len)]
    generated = list(seed)

    with torch.no_grad():
        for _ in range(GENERATE_LENGTH):
            context = torch.tensor([generated[-seq_len:]], dtype=torch.long, device=device)
            logits, _ = model(context)
            next_logits = logits[0, -1, :]
            next_idx = sample_next_token(next_logits, TEMPERATURE)
            generated.append(next_idx)

    return [idx2tok[i] for i in generated[seq_len:]]  # drop the random seed itself


def tokens_to_midi(tokens, out_path, bpm=100):
    """Convert 'PITCH_DURATION' tokens (e.g. 'C#5_0.5', 'E4.G4_1.0', 'REST_0.5')
    back into a music21 Stream and write it out as a .mid file."""
    melody = stream.Stream()
    melody.append(tempo.MetronomeMark(number=bpm))

    for tok in tokens:
        pitch_part, dur_str = tok.rsplit("_", 1)
        dur = float(dur_str)

        if pitch_part == "REST":
            el = note.Rest()
        elif "." in pitch_part:
            el = chord.Chord(pitch_part.split("."))
        else:
            el = note.Note(pitch_part)

        el.duration.quarterLength = dur
        melody.append(el)

    melody.write("midi", fp=str(out_path))
    return out_path


def main():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    tok2idx, idx2tok = checkpoint["tok2idx"], checkpoint["idx2tok"]
    seq_len = checkpoint["seq_len"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicLSTM(checkpoint["vocab_size"]).to(device)
    model.load_state_dict(checkpoint["model_state"])

    print(f"Generating {GENERATE_LENGTH} note events (temperature={TEMPERATURE})...")
    tokens = generate_token_sequence(model, tok2idx, idx2tok, seq_len, device)
    print("Generated tokens:", tokens[:12], "...")

    out_path = OUTPUT_DIR / "generated_melody.mid"
    tokens_to_midi(tokens, out_path)
    print(f"\nSaved MIDI to {out_path}")


if __name__ == "__main__":
    main()