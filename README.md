# 🎵 AI Mini Projects – Code Alpha Internship

## Task 3: Music Generation with AI

This repository contains an AI-focused Python project built during my internship with Code Alpha. It's a beginner-friendly deep learning pipeline that explores sequence modeling and generative AI for music.

## ✅ Project Included

### 🎵 Task 3: Music Generation with AI

- Trains on classical music data (Bach chorales, via `music21`'s built-in corpus)
- Preprocesses MIDI/score data into note-and-duration sequences
- Builds and trains an **LSTM** neural network to learn musical patterns
- Generates brand-new note sequences and converts them into a playable **MIDI** file
- Built using Python, **PyTorch**, and **music21**

## 📂 File Structure

```
task3-music-generation-with-ai/
├── 01_preprocess.py
├── 02_model.py
├── 03_train.py
├── 04_generate.py
├── 05_midi_to_audio.py
├── requirements.txt
└── README.md
```

## 🚀 How to Run

Make sure you have **Python 3.8+** installed.

**1. Clone this repo**
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

**2. Install required libraries**
```bash
pip install -r requirements.txt
```
> No dataset download needed — `music21` ships with 433 Bach chorales
> built into the library itself.

**3. Run the pipeline, in order**
```bash
python 01_preprocess.py    # parses Bach chorales -> data/notes.pkl
python 03_train.py         # trains the LSTM -> models/music_lstm.pt
python 04_generate.py      # generates a new melody -> output/generated_melody.mid
```

Open the resulting `.mid` file in any DAW (GarageBand, Ableton) or notation
software (MuseScore) to listen to or view the generated music. Re-running
`04_generate.py` produces a different melody each time.

*(Optional)* Render the `.mid` to a playable `.wav` with `05_midi_to_audio.py` — requires a local FluidSynth install, see the script for setup steps.
