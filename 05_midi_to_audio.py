import shutil
import subprocess
import sys
from pathlib import Path


def midi_to_wav(midi_path: str, soundfont_path: str, out_path: str | None = None) -> str:
    if shutil.which("fluidsynth") is None:
        raise RuntimeError(
            "fluidsynth is not installed. Install it first, e.g.:\n"
            "  Linux:  sudo apt-get install fluidsynth\n"
            "  macOS:  brew install fluidsynth\n"
            "Then download a free .sf2 SoundFont (e.g. FluidR3_GM.sf2)."
        )
    midi_path = Path(midi_path)
    out_path = Path(out_path) if out_path else midi_path.with_suffix(".wav")

    subprocess.run(
        ["fluidsynth", "-ni", soundfont_path, str(midi_path), "-F", str(out_path), "-r", "44100"],
        check=True,
    )
    return str(out_path)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python 05_midi_to_audio.py <input.mid> <soundfont.sf2> [output.wav]")
        sys.exit(1)
    result = midi_to_wav(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    print(f"Saved audio to {result}")