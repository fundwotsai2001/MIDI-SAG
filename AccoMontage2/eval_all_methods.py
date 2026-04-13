"""
Evaluate all key detection methods on:
  1. /data/home/fundwotsai/MIDI-SAG/lyrics2melody/share
     (ground truth from filename: song-artist-tempo-KEY.mid)
  2. /data/home/fundwotsai/MIDI-SAG/lyrics2melody/generated_midi
     (ground truth: C major for all files)
"""
import os, sys, glob, warnings, importlib
import numpy as np
from collections import defaultdict

warnings.filterwarnings('ignore')

SHARE_DIR     = '/data/home/fundwotsai/MIDI-SAG/lyrics2melody/share'
GENERATED_DIR = '/data/home/fundwotsai/MIDI-SAG/lyrics2melody/generated_midi'

sys.path.insert(0, '/data/home/fundwotsai/MIDI-SAG/AccoMontage2')

# ---------------------------------------------------------------------------
# ENHARMONIC normalisation
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
    'bB': 'Bb', 'bE': 'Eb', 'bA': 'Ab', 'bD': 'Db', 'bG': 'Gb',
    '#F': 'Gb', '#C': 'Db', '#G': 'Ab', '#D': 'Eb', '#A': 'Bb',
}
PC_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

def normalise(result):
    pc = ENHARMONIC.get(result['key'], result['key'])
    return pc, result['mode']

def semitone_dist(a, b):
    if a not in PC_NAMES or b not in PC_NAMES: return -1
    d = abs(PC_NAMES.index(a) - PC_NAMES.index(b))
    return min(d, 12 - d)

def parse_filename_key(stem):
    parts = stem.split('-')
    if len(parts) < 2: return None, None
    raw = parts[-1].strip()
    if raw.endswith('m'): mode, raw_pc = 'minor', raw[:-1]
    else:                  mode, raw_pc = 'major', raw
    pc = ENHARMONIC.get(raw_pc)
    return (pc, mode) if pc else (None, None)

# ---------------------------------------------------------------------------
# Method 1: KS (duration-weighted Krumhansl-Schmuckler)
# ---------------------------------------------------------------------------
import mido

def ks_key_detect(midi_path):
    KS_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    KS_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
    NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    SHARP_TO_FLAT = {'C#':'Db','D#':'Eb','F#':'Gb','G#':'Ab','A#':'Bb'}

    mid = mido.MidiFile(midi_path); tpb = mid.ticks_per_beat; tempo = 500000
    pc_duration = np.zeros(12)
    for track in mid.tracks:
        active = {}; abs_sec = 0.0
        for msg in track:
            if msg.time > 0: abs_sec += mido.tick2second(msg.time, tpb, tempo)
            if msg.type == 'set_tempo': tempo = msg.tempo
            elif msg.type == 'note_on' and msg.velocity > 0:
                active[(msg.channel, msg.note)] = abs_sec
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                k = (msg.channel, msg.note)
                if k in active: pc_duration[msg.note % 12] += abs_sec - active.pop(k)
    if pc_duration.sum() == 0: return {'key':'C','mode':'major','confidence':0.0}
    v = pc_duration - pc_duration.mean()
    best_r, best_key, best_mode = -2.0, 'C', 'major'
    for root in range(12):
        maj = np.roll(KS_MAJOR, root); maj -= maj.mean()
        r = np.dot(v, maj) / (np.linalg.norm(v)*np.linalg.norm(maj)+1e-9)
        if r > best_r: best_r, best_key, best_mode = r, NOTE_NAMES[root], 'major'
        min_ = np.roll(KS_MINOR, root); min_ -= min_.mean()
        r = np.dot(v, min_) / (np.linalg.norm(v)*np.linalg.norm(min_)+1e-9)
        if r > best_r: best_r, best_key, best_mode = r, NOTE_NAMES[root], 'minor'
    best_key = SHARP_TO_FLAT.get(best_key, best_key)
    return {'key':best_key,'mode':best_mode,'confidence':float(best_r)}

# ---------------------------------------------------------------------------
# Method 2: Improved KS + leading-tone
# ---------------------------------------------------------------------------
def _parse_midi_notes(midi_path):
    mid = mido.MidiFile(midi_path); tpb = mid.ticks_per_beat; tempo = 500000; notes = []
    for track in mid.tracks:
        active = {}; t = 0.0
        for msg in track:
            if msg.time > 0: t += mido.tick2second(msg.time, tpb, tempo)
            if msg.type == 'set_tempo': tempo = msg.tempo
            elif msg.type == 'note_on' and msg.velocity > 0: active[(msg.channel,msg.note)] = t
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                k = (msg.channel, msg.note)
                if k in active: notes.append((active.pop(k), t, msg.note))
    notes.sort(); return notes

def improved_key_detect(midi_path):
    KS_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    KS_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
    NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    SHARP_TO_FLAT = {'C#':'Db','D#':'Eb','F#':'Gb','G#':'Ab','A#':'Bb'}

    notes = _parse_midi_notes(midi_path)
    if not notes: return {'key':'C','mode':'major','confidence':0.0}
    pc_dur = np.zeros(12)
    for onset, offset, pitch in notes: pc_dur[pitch%12] += (offset-onset)
    if pc_dur.sum() == 0: return {'key':'C','mode':'major','confidence':0.0}
    pc_lead = np.zeros(12)
    for i in range(1, len(notes)):
        if (notes[i][2]-notes[i-1][2])%12 == 1:
            pc_lead[notes[i][2]%12] += notes[i][1]-notes[i][0]
    def norm(x): s=x.sum(); return x/s if s>0 else np.ones(12)/12
    v = (0.70*norm(pc_dur)+0.30*norm(pc_lead)) if pc_lead.sum()>0 else norm(pc_dur)
    v -= v.mean()
    scores = {}
    for root in range(12):
        maj = np.roll(KS_MAJOR,root); maj -= maj.mean()
        scores[(root,'major')] = float(np.dot(v,maj)/(np.linalg.norm(v)*np.linalg.norm(maj)+1e-9))
        min_ = np.roll(KS_MINOR,root); min_ -= min_.mean()
        scores[(root,'minor')] = float(np.dot(v,min_)/(np.linalg.norm(v)*np.linalg.norm(min_)+1e-9))
    last_pc = notes[-1][2]%12
    for root in range(12):
        if root == last_pc:
            scores[(root,'major')] += 0.08; scores[(root,'minor')] += 0.08
    best_root, best_mode = max(scores, key=scores.get)
    key_name = SHARP_TO_FLAT.get(NOTE_NAMES[best_root], NOTE_NAMES[best_root])
    return {'key':key_name,'mode':best_mode,'confidence':float(scores[(best_root,best_mode)])}

# ---------------------------------------------------------------------------
# Method 3: demo_utils get_detailed_key_analysis (REMI token count-based)
# ---------------------------------------------------------------------------
from miditok import REMI, TokenizerConfig
from symusic import Score as SymScore
_tokenizer = REMI(TokenizerConfig(num_velocities=16, use_chords=False, use_programs=False))

def demo_utils_key_detect(midi_path):
    du = importlib.import_module('demo_utils')
    midi_obj = SymScore(midi_path)
    tokens = _tokenizer(midi_obj)
    if len(tokens) == 1: tokens = tokens[0]
    result = du.get_detailed_key_analysis(tokens.tokens)
    return {'key': result['key'], 'mode': result['mode'], 'confidence': result['confidence']}

# ---------------------------------------------------------------------------
# Method 4-6: Partitura profiles
# ---------------------------------------------------------------------------
import partitura as _pt
import partitura.musicanalysis.key_identification as _ki

def _partitura_detect(midi_path, profile):
    score = _pt.load_score(midi_path)
    na = score[0].note_array()
    result = _ki.ks_kid(na, key_profiles=profile).strip("'")
    mode, pc_raw = ('minor', result[:-1]) if result.endswith('m') else ('major', result)
    return {'key': pc_raw, 'mode': mode, 'confidence': 1.0}

def partitura_ks_detect(midi_path):   return _partitura_detect(midi_path, 'ks')
def partitura_kp_detect(midi_path):   return _partitura_detect(midi_path, 'kp')
def partitura_cmbs_detect(midi_path): return _partitura_detect(midi_path, 'cmbs')

# ---------------------------------------------------------------------------
# Evaluation engine
# ---------------------------------------------------------------------------
ALL_METHODS = [
    ('demo_utils (scale-fit+tonic bonus)', demo_utils_key_detect),
    ('KS (duration-weighted)',             ks_key_detect),
    ('Improved KS+leading-tone',           improved_key_detect),
    ('Partitura KS',                       partitura_ks_detect),
    ('Partitura Kostka-Payne',             partitura_kp_detect),
    ('Partitura Bellman-Budge',            partitura_cmbs_detect),
]

def evaluate(name, detect_fn, midi_paths, gt_fn):
    """gt_fn(midi_path) -> (gt_pc, gt_mode) or (None, None) to skip."""
    total = key_ok = mode_ok = full_ok = errors = skipped = 0
    wrong = defaultdict(int)
    interval_err = defaultdict(int)

    for path in midi_paths:
        gt_pc, gt_mode = gt_fn(path)
        if gt_pc is None: skipped += 1; continue
        total += 1
        try:
            result = detect_fn(path)
        except Exception as e:
            errors += 1; total -= 1
            print(f"  ERROR {os.path.basename(path)}: {e}")
            continue
        pred_pc, pred_mode = normalise(result)
        pc_ok = pred_pc == gt_pc
        mo_ok = pred_mode == gt_mode
        key_ok  += int(pc_ok)
        mode_ok += int(mo_ok)
        full_ok += int(pc_ok and mo_ok)
        if not pc_ok:
            wrong[f"{pred_pc} {pred_mode}"] += 1
            d = semitone_dist(gt_pc, pred_pc)
            interval_err[d] += 1

    n = total
    print(f"\n{'='*58}")
    print(f"METHOD : {name}")
    print(f"{'='*58}")
    print(f"  Files evaluated : {n}   Skipped: {skipped}   Errors: {errors}")
    if n:
        print(f"  Pitch-class     : {key_ok}/{n} = {100*key_ok/n:.1f}%")
        print(f"  Mode            : {mode_ok}/{n} = {100*mode_ok/n:.1f}%")
        print(f"  Full key+mode   : {full_ok}/{n} = {100*full_ok/n:.1f}%")
    if wrong:
        print(f"  Top wrong preds :")
        for k, c in sorted(wrong.items(), key=lambda x:-x[1])[:6]:
            print(f"    {k:<15}: {c}")
    if interval_err:
        print(f"  Semitone error distribution:")
        for d, c in sorted(interval_err.items(), key=lambda x:-x[1])[:5]:
            print(f"    {d:2d} st : {c}")
    print('='*58)
    return full_ok, n

def print_summary(label, results):
    print(f"\n{'#'*58}")
    print(f"SUMMARY — {label}")
    print(f"{'#'*58}")
    for name, full_ok, n in results:
        pct = f"{100*full_ok/n:.1f}%" if n else "N/A"
        print(f"  {name:<40} {full_ok}/{n} = {pct}")
    print('#'*58)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # --- Dataset 1: share (GT from filename) ---
    share_files = sorted(glob.glob(os.path.join(SHARE_DIR, '*.mid')))
    def share_gt(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return parse_filename_key(stem)

    print(f"\n{'#'*58}")
    print(f"DATASET 1: share ({len(share_files)} files, GT from filename)")
    print(f"{'#'*58}")
    share_results = []
    for name, fn in ALL_METHODS:
        print(f"\nRunning [{name}] on share ...")
        full_ok, n = evaluate(name, fn, share_files, share_gt)
        share_results.append((name, full_ok, n))
    print_summary('share', share_results)

    # --- Dataset 2: generated_midi (GT = C major) ---
    gen_files = sorted(glob.glob(os.path.join(GENERATED_DIR, 'lyrics_*/sample01.mid')))
    def gen_gt(path): return 'C', 'major'

    print(f"\n{'#'*58}")
    print(f"DATASET 2: generated_midi ({len(gen_files)} files, GT = C major)")
    print(f"{'#'*58}")
    gen_results = []
    for name, fn in ALL_METHODS:
        print(f"\nRunning [{name}] on generated_midi ...")
        full_ok, n = evaluate(name, fn, gen_files, gen_gt)
        gen_results.append((name, full_ok, n))
    print_summary('generated_midi', gen_results)
