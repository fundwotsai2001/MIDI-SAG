"""
Convert a MIDI score file to SoulX-Singer target metadata JSON.
Lyrics are read from the MIDI's embedded lyric events.

Usage (run from SoulX-Singer/ root):
    python midi2json.py <midi_path> <output_json_path> [--language Mandarin]
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess.tools.midi_parser import MidiParser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("midi_path", type=str)
    parser.add_argument("output_json", type=str)
    parser.add_argument("--language", type=str, default="Mandarin",
                        choices=["Mandarin", "English", "Cantonese"])
    args = parser.parse_args()

    print(f"[midi2json] Reading: {args.midi_path}")
    print(f"[midi2json] Language: {args.language}")

    midi_parser = MidiParser(
        rmvpe_model_path="pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt",
        device="cpu",   # F0 not needed for score-based inference
    )
    midi_parser.midi2meta(
        midi_path=args.midi_path,
        meta_path=args.output_json,
        vocal_file=None,        # no reference audio → f0 left empty (score mode)
        language=args.language,
    )
    print(f"[midi2json] Saved: {args.output_json}")


if __name__ == "__main__":
    main()
