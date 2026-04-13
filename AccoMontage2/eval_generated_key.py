"""
Evaluate key detection methods on generated_midi/lyrics_N/sample01.mid files.
Ground truth: C major for all files.
"""
import os, sys
import glob
import warnings
import numpy as np

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# KS key detect (duration-weighted Krumhansl-Schmuckler)
# ---------------------------------------------------------------------------
import mido

def ks_key_detect(midi_path):
    KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                         2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                         2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']
    SHARP_TO_FLAT = {'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb', 'G#': 'Ab', 'A#': 'Bb'}

    mid = mido.MidiFile(midi_path)
    tpb = mid.ticks_per_beat
    tempo = 500000
    pc_duration = np.zeros(12)
    for track in mid.tracks:
        active = {}; abs_sec = 0.0
        for msg in track:
            if msg.time > 0:
                abs_sec += mido.tick2second(msg.time, tpb, tempo)
            if msg.type == 'set_tempo': tempo = msg.tempo
            elif msg.type == 'note_on' and msg.velocity > 0:
                active[(msg.channel, msg.note)] = abs_sec
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                k = (msg.channel, msg.note)
                if k in active:
                    pc_duration[msg.note % 12] += abs_sec - active.pop(k)

    if pc_duration.sum() == 0:
        return {'key': 'C', 'mode': 'major', 'confidence': 0.0}
    v = pc_duration - pc_duration.mean()
    best_r, best_key, best_mode = -2.0, 'C', 'major'
    for root in range(12):
        maj = np.roll(KS_MAJOR, root); maj -= maj.mean()
        r = np.dot(v, maj) / (np.linalg.norm(v) * np.linalg.norm(maj) + 1e-9)
        if r > best_r: best_r, best_key, best_mode = r, NOTE_NAMES[root], 'major'
        min_ = np.roll(KS_MINOR, root); min_ -= min_.mean()
        r = np.dot(v, min_) / (np.linalg.norm(v) * np.linalg.norm(min_) + 1e-9)
        if r > best_r: best_r, best_key, best_mode = r, NOTE_NAMES[root], 'minor'
    best_key = SHARP_TO_FLAT.get(best_key, best_key)
    return {'key': best_key, 'mode': best_mode, 'confidence': float(best_r)}


# ---------------------------------------------------------------------------
# Improved KS + leading-tone
# ---------------------------------------------------------------------------
def _parse_midi_notes(midi_path):
    mid = mido.MidiFile(midi_path)
    tpb = mid.ticks_per_beat; tempo = 500000; notes = []
    for track in mid.tracks:
        active = {}; t = 0.0
        for msg in track:
            if msg.time > 0: t += mido.tick2second(msg.time, tpb, tempo)
            if msg.type == 'set_tempo': tempo = msg.tempo
            elif msg.type == 'note_on' and msg.velocity > 0:
                active[(msg.channel, msg.note)] = t
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                k = (msg.channel, msg.note)
                if k in active: notes.append((active.pop(k), t, msg.note))
    notes.sort(); return notes

def improved_key_detect(midi_path):
    KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                         2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                         2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    NOTE_NAMES  = ['C', 'C#', 'D', 'D#', 'E', 'F',
                   'F#', 'G', 'G#', 'A', 'A#', 'B']
    SHARP_TO_FLAT = {'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb', 'G#': 'Ab', 'A#': 'Bb'}

    notes = _parse_midi_notes(midi_path)
    if not notes: return {'key': 'C', 'mode': 'major', 'confidence': 0.0}

    pc_dur = np.zeros(12)
    for onset, offset, pitch in notes:
        pc_dur[pitch % 12] += (offset - onset)
    if pc_dur.sum() == 0: return {'key': 'C', 'mode': 'major', 'confidence': 0.0}

    pc_lead = np.zeros(12)
    for i in range(1, len(notes)):
        interval = (notes[i][2] - notes[i - 1][2]) % 12
        if interval == 1:
            dur = notes[i][1] - notes[i][0]
            pc_lead[notes[i][2] % 12] += dur

    def norm(x): s = x.sum(); return x / s if s > 0 else np.ones(12) / 12

    v = (0.70 * norm(pc_dur) + 0.30 * norm(pc_lead)) if pc_lead.sum() > 0 else norm(pc_dur)
    v -= v.mean()

    scores = {}
    for root in range(12):
        maj = np.roll(KS_MAJOR, root); maj -= maj.mean()
        scores[(root, 'major')] = float(np.dot(v, maj) / (np.linalg.norm(v) * np.linalg.norm(maj) + 1e-9))
        min_ = np.roll(KS_MINOR, root); min_ -= min_.mean()
        scores[(root, 'minor')] = float(np.dot(v, min_) / (np.linalg.norm(v) * np.linalg.norm(min_) + 1e-9))

    last_pc = notes[-1][2] % 12
    for root in range(12):
        if root == last_pc:
            scores[(root, 'major')] += 0.08
            scores[(root, 'minor')] += 0.08

    best_root, best_mode = max(scores, key=scores.get)
    best_r = scores[(best_root, best_mode)]
    key_name = SHARP_TO_FLAT.get(NOTE_NAMES[best_root], NOTE_NAMES[best_root])
    return {'key': key_name, 'mode': best_mode, 'confidence': float(best_r)}


# ---------------------------------------------------------------------------
# demo_utils: get_detailed_key_analysis (REMI token-based, count + tonic bonus)
# ---------------------------------------------------------------------------
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../AccoMontage2')
_sys.path.insert(0, '/data/home/fundwotsai/MIDI-SAG/AccoMontage2')

from miditok import REMI, TokenizerConfig
from symusic import Score as SymScore

_tokenizer = REMI(TokenizerConfig(num_velocities=16, use_chords=False, use_programs=False))

def demo_utils_key_detect(midi_path):
    import importlib, sys
    # lazy import to avoid circular issues
    du = importlib.import_module('demo_utils')
    midi_obj = SymScore(midi_path)
    tokens = _tokenizer(midi_obj)
    if len(tokens) == 1:
        tokens = tokens[0]
    result = du.get_detailed_key_analysis(tokens.tokens)
    return {'key': result['key'], 'mode': result['mode'], 'confidence': result['confidence']}


# ---------------------------------------------------------------------------
# Partitura wrappers
# ---------------------------------------------------------------------------
import partitura as _pt
import partitura.musicanalysis.key_identification as _ki

def _partitura_detect(midi_path, profile):
    score = _pt.load_score(midi_path)
    na = score[0].note_array()
    result = _ki.ks_kid(na, key_profiles=profile).strip("'")
    if result.endswith('m'):
        mode, pc_raw = 'minor', result[:-1]
    else:
        mode, pc_raw = 'major', result
    return {'key': pc_raw, 'mode': mode, 'confidence': 1.0}

def partitura_ks_detect(midi_path):   return _partitura_detect(midi_path, 'ks')
def partitura_kp_detect(midi_path):   return _partitura_detect(midi_path, 'kp')
def partitura_cmbs_detect(midi_path): return _partitura_detect(midi_path, 'cmbs')


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
ENHARMONIC = {
    'C': 'C', 'B#': 'C',
    'Db': 'Db', 'C#': 'Db',
    'D': 'D',
    'Eb': 'Eb', 'D#': 'Eb',
    'E': 'E', 'Fb': 'E',
    'F': 'F', 'E#': 'F',
    'Gb': 'Gb', 'F#': 'Gb',
    'G': 'G',
    'Ab': 'Ab', 'G#': 'Ab',
    'A': 'A',
    'Bb': 'Bb', 'A#': 'Bb',
    'B': 'B', 'Cb': 'B',
}

def normalise(result):
    pc = ENHARMONIC.get(result['key'], result['key'])
    return pc, result['mode']

GT_KEY  = 'C'
GT_MODE = 'major'

GENERATED_DIR = '/data/home/fundwotsai/MIDI-SAG/lyrics2melody/generated_midi'

def run_eval(name, detect_fn, midi_files):
    correct_key  = 0
    correct_mode = 0
    correct_full = 0
    errors       = 0
    wrong_keys   = {}
    total = len(midi_files)

    for path in midi_files:
        try:
            result = detect_fn(path)
        except Exception as e:
            errors += 1
            print(f"  ERROR {os.path.basename(os.path.dirname(path))}: {e}")
            continue

        pred_pc, pred_mode = normalise(result)
        pc_ok   = pred_pc   == GT_KEY
        mode_ok = pred_mode == GT_MODE

        correct_key  += int(pc_ok)
        correct_mode += int(mode_ok)
        correct_full += int(pc_ok and mode_ok)
        if not (pc_ok and mode_ok):
            label = f"{pred_pc} {pred_mode}"
            wrong_keys[label] = wrong_keys.get(label, 0) + 1

    n = total - errors
    print(f"\n{'='*55}")
    print(f"METHOD : {name}")
    print(f"{'='*55}")
    print(f"  Total files  : {total}   Errors: {errors}")
    if n:
        print(f"  Pitch-class  : {correct_key}/{n}  = {100*correct_key/n:.1f}%")
        print(f"  Mode         : {correct_mode}/{n}  = {100*correct_mode/n:.1f}%")
        print(f"  Full (key+mode): {correct_full}/{n} = {100*correct_full/n:.1f}%")
    if wrong_keys:
        print(f"  Wrong predictions (pred : count):")
        for k, c in sorted(wrong_keys.items(), key=lambda x: -x[1]):
            print(f"    {k:<15}: {c}")
    print('='*55)
    return correct_full, n

if __name__ == '__main__':
    midi_files = sorted(
        glob.glob(os.path.join(GENERATED_DIR, 'lyrics_*/sample01.mid'))
    )
    print(f"Found {len(midi_files)} MIDI files.")
    print(f"Ground truth: {GT_KEY} {GT_MODE}\n")

    methods = [
        ('demo_utils get_detailed_key_analysis', demo_utils_key_detect),
        ('KS (ks_key_detect)',                   ks_key_detect),
        ('Improved KS+leading-tone',             improved_key_detect),
        ('Partitura KS (ks)',                    partitura_ks_detect),
        ('Partitura Kostka-Payne',               partitura_kp_detect),
        ('Partitura Bellman-Budge',              partitura_cmbs_detect),
    ]

    summary = []
    for name, fn in methods:
        print(f"Running: {name} ...")
        correct, total = run_eval(name, fn, midi_files)
        if total:
            summary.append((name, correct, total))

    print(f"\n{'='*55}")
    print("SUMMARY")
    print(f"{'='*55}")
    for name, correct, total in summary:
        print(f"  {name:<35} {correct}/{total} = {100*correct/total:.1f}%")
    print('='*55)
