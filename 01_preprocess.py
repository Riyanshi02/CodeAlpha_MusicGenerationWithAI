import pickle
from pathlib import Path
from music21 import corpus, note, chord

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

N_PIECES = 60  # how many chorales to use (keep training fast on CPU)


def extract_event_sequence(score):
    """Extract the top (soprano) line of a chorale as a sequence of
    'PITCH_DURATION' string tokens, e.g. 'C#5_0.5'. Chords (rare in the
    soprano line, but possible) are encoded as dot-joined pitch names."""
    part = score.parts[0]  # soprano is conventionally the first part
    events = []
    for el in part.flatten().notesAndRests:
        if isinstance(el, note.Note):
            pitch_str = el.pitch.nameWithOctave
        elif isinstance(el, chord.Chord):
            pitch_str = ".".join(p.nameWithOctave for p in el.pitches)
        else:  # Rest
            pitch_str = "REST"
        dur = round(float(el.duration.quarterLength), 2)
        events.append(f"{pitch_str}_{dur}")
    return events


def main():
    bach_paths = corpus.getComposer("bach")[:N_PIECES]
    print(f"Parsing {len(bach_paths)} Bach chorales from the music21 corpus...")

    all_sequences = []
    for i, path in enumerate(bach_paths):
        try:
            score = corpus.parse(path)
            events = extract_event_sequence(score)
            if len(events) >= 16:  # skip fragments too short to train on
                all_sequences.append(events)
        except Exception as e:
            print(f"  skipped {path.name}: {e}")
        if (i + 1) % 10 == 0:
            print(f"  parsed {i + 1}/{len(bach_paths)}")

    # Build vocabulary over all tokens seen
    vocab = sorted({tok for seq in all_sequences for tok in seq})
    tok2idx = {tok: i for i, tok in enumerate(vocab)}
    idx2tok = {i: tok for tok, i in tok2idx.items()}

    print(f"\nPieces kept: {len(all_sequences)}")
    print(f"Vocabulary size (unique pitch+duration tokens): {len(vocab)}")
    total_notes = sum(len(s) for s in all_sequences)
    print(f"Total note events: {total_notes}")

    with open(DATA_DIR / "notes.pkl", "wb") as f:
        pickle.dump(
            {"sequences": all_sequences, "tok2idx": tok2idx, "idx2tok": idx2tok},
            f,
        )
    print(f"\nSaved to {DATA_DIR / 'notes.pkl'}")


if __name__ == "__main__":
    main()