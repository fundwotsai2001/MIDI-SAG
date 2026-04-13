"""
Compare key detection methods against ground-truth keys embedded in MIDI filenames.

Methods tested:
  1. ks_key_detect      — from demo_SOME.py   (duration-weighted Pearson / KS profiles)
  2. get_detailed_key_analysis — from demo_utils.py  (count-based scale-fit + tonic bonus)
  3. improved_key_detect — new method combining:
       a) Global duration-weighted KS profiles
       b) Phrase-ending cadence histogram (notes before silences)
       c) Positionally-weighted histogram (end-of-piece bias)

Filename format: song-artist-tempo-KEY.mid
  KEY examples: C, Db, F#, bB, bE, #F, Gm, Abm, Em, ...
"""

import os
import sys
import numpy as np
import mido
from collections import defaultdict

# Make demo_utils importable
sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Method 1: ks_key_detect (verbatim from demo_SOME.py)
# ---------------------------------------------------------------------------
def ks_key_detect(midi_path):
    """Duration-weighted Krumhansl-Schmuckler key detection."""
    KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                         2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                         2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']

    mid = mido.MidiFile(midi_path)
    tpb = mid.ticks_per_beat
    tempo = 500000
    pc_duration = np.zeros(12)

    for track in mid.tracks:
        active = {}
        abs_sec = 0.0
        for msg in track:
            if msg.time > 0:
                abs_sec += mido.tick2second(msg.time, tpb, tempo)
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            elif msg.type == 'note_on' and msg.velocity > 0:
                active[(msg.channel, msg.note)] = abs_sec
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                k = (msg.channel, msg.note)
                if k in active:
                    dur = abs_sec - active.pop(k)
                    pc_duration[msg.note % 12] += dur

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

    SHARP_TO_FLAT = {'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb', 'G#': 'Ab', 'A#': 'Bb'}
    best_key = SHARP_TO_FLAT.get(best_key, best_key)
    return {'key': best_key, 'mode': best_mode, 'confidence': float(best_r)}


# ---------------------------------------------------------------------------
# Method 3: improved_key_detect
#
# Three changes on top of standard KS, all using only melodic content:
#
#  A) Leading-tone resolution histogram (ascending half-step targets)
#     In a major key, the 7th scale degree resolves up by a half step to the
#     tonic (e.g. F#→G in G major).  In natural minor the 7th is a whole step
#     below the tonic (D→E), so no half-step resolution TO the tonic exists.
#     Counting the TARGET of every ascending half-step interval gives a
#     histogram that peaks at the tonic in major keys and directly distinguishes
#     G major (peak at G) from E natural minor (no half-step approach to E).
#
#  B) Combined KS input: 70% standard duration-weighted + 30% leading-tone
#     histogram.  This feeds into the same Pearson KS computation.
#
#  C) Last-note tonic anchor (+0.08 bonus):
#     58% of songs in this corpus end on the tonic; a small bonus breaks
#     near-ties between a key and its relative major/minor.
# ---------------------------------------------------------------------------
def _parse_midi_notes(midi_path):
    """Return sorted list of (onset_sec, offset_sec, pitch)."""
    mid = mido.MidiFile(midi_path)
    tpb = mid.ticks_per_beat
    tempo = 500000
    notes = []
    for track in mid.tracks:
        active = {}
        t = 0.0
        for msg in track:
            if msg.time > 0:
                t += mido.tick2second(msg.time, tpb, tempo)
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            elif msg.type == 'note_on' and msg.velocity > 0:
                active[(msg.channel, msg.note)] = t
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                k = (msg.channel, msg.note)
                if k in active:
                    notes.append((active.pop(k), t, msg.note))
    notes.sort()
    return notes


def improved_key_detect(midi_path):
    KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                         2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                         2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    NOTE_NAMES  = ['C', 'C#', 'D', 'D#', 'E', 'F',
                   'F#', 'G', 'G#', 'A', 'A#', 'B']
    SHARP_TO_FLAT = {'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb', 'G#': 'Ab', 'A#': 'Bb'}

    notes = _parse_midi_notes(midi_path)
    if not notes:
        return {'key': 'C', 'mode': 'major', 'confidence': 0.0}

    # A) Duration-weighted pitch-class histogram (standard KS input)
    pc_dur = np.zeros(12)
    for onset, offset, pitch in notes:
        pc_dur[pitch % 12] += (offset - onset)
    if pc_dur.sum() == 0:
        return {'key': 'C', 'mode': 'major', 'confidence': 0.0}

    # B) Leading-tone resolution histogram
    # For each consecutive pair, if the interval is an ascending half step
    # (semitones mod 12 == 1), the upper (target) note gets a vote weighted
    # by its own duration — it is being resolved to as a stable tone.
    pc_lead = np.zeros(12)
    for i in range(1, len(notes)):
        interval = (notes[i][2] - notes[i - 1][2]) % 12
        if interval == 1:                          # ascending half step
            dur = notes[i][1] - notes[i][0]
            pc_lead[notes[i][2] % 12] += dur

    def norm(x):
        s = x.sum()
        return x / s if s > 0 else np.ones(12) / 12

    # Combine: fall back to pure KS when no chromatic motion found
    if pc_lead.sum() > 0:
        v = 0.70 * norm(pc_dur) + 0.30 * norm(pc_lead)
    else:
        v = norm(pc_dur)
    v -= v.mean()

    # KS Pearson correlation for all 24 keys
    scores = {}
    for root in range(12):
        maj = np.roll(KS_MAJOR, root); maj -= maj.mean()
        scores[(root, 'major')] = float(
            np.dot(v, maj) / (np.linalg.norm(v) * np.linalg.norm(maj) + 1e-9))
        min_ = np.roll(KS_MINOR, root); min_ -= min_.mean()
        scores[(root, 'minor')] = float(
            np.dot(v, min_) / (np.linalg.norm(v) * np.linalg.norm(min_) + 1e-9))

    # C) Last-note anchor: small bonus to break near-ties
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
# Method 2: get_detailed_key_analysis from demo_utils.py (via REMI tokens)
# ---------------------------------------------------------------------------
from demo_utils import get_detailed_key_analysis

from miditok import REMI, TokenizerConfig
from symusic import Score as SymScore

_tokenizer = REMI(TokenizerConfig(num_velocities=16, use_chords=False, use_programs=False))

def demo_utils_key_detect(midi_path):
    """Count-based scale-fit detection using demo_utils.get_detailed_key_analysis."""
    midi_obj = SymScore(midi_path)
    tokens = _tokenizer(midi_obj)
    if len(tokens) == 1:
        tokens = tokens[0]
    result = get_detailed_key_analysis(tokens.tokens)
    return result


# ---------------------------------------------------------------------------
# music21 and partitura wrappers
# ---------------------------------------------------------------------------
import warnings as _warnings
_warnings.filterwarnings('ignore')

import partitura as _pt
import partitura.musicanalysis.key_identification as _ki

def _partitura_detect(midi_path, profile):
    score = _pt.load_score(midi_path)
    na = score[0].note_array()
    result = _ki.ks_kid(na, key_profiles=profile)   # e.g. "Gm" or "G"
    result = result.strip("'")
    if result.endswith('m'):
        mode, pc_raw = 'minor', result[:-1]
    else:
        mode, pc_raw = 'major', result
    return {'key': pc_raw, 'mode': mode, 'confidence': 1.0}

def partitura_ks_detect(midi_path):
    """Partitura — Krumhansl-Kessler profiles (same as standard KS)."""
    return _partitura_detect(midi_path, 'ks')

def partitura_kp_detect(midi_path):
    """Partitura — Kostka-Payne profiles."""
    return _partitura_detect(midi_path, 'kp')

def partitura_cmbs_detect(midi_path):
    """Partitura — Bellman-Budge (Temperley) profiles."""
    return _partitura_detect(midi_path, 'cmbs')


from music21 import converter as _m21conv

def music21_ks_detect(midi_path):
    """music21 — Krumhansl-Schmuckler key analysis."""
    s = _m21conv.parse(midi_path)
    k = s.analyze('key')
    return {'key': k.tonic.name, 'mode': k.mode,
            'confidence': float(k.correlationCoefficient)}

def music21_aarden_detect(midi_path):
    """music21 — Aarden-Essen key profiles."""
    s = _m21conv.parse(midi_path)
    k = s.analyze('AardenEssen')
    return {'key': k.tonic.name, 'mode': k.mode,
            'confidence': float(k.correlationCoefficient)}


# ---------------------------------------------------------------------------
# Key normalisation helpers
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
    # b-prefix notation common in Chinese MIDI datasets
    'bB': 'Bb', 'bE': 'Eb', 'bA': 'Ab', 'bD': 'Db', 'bG': 'Gb',
    # #-prefix notation
    '#F': 'Gb', '#C': 'Db', '#G': 'Ab', '#D': 'Eb', '#A': 'Bb',
}

PC_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']


def parse_filename_key(stem: str):
    """Return (canonical_pc, mode) from filename stem, or (None, None) if unrecognised."""
    parts = stem.split('-')
    if len(parts) < 2:
        return None, None
    raw = parts[-1].strip()
    if raw.endswith('m'):
        mode, raw_pc = 'minor', raw[:-1]
    else:
        mode, raw_pc = 'major', raw
    pc = ENHARMONIC.get(raw_pc)
    return (pc, mode) if pc else (None, None)


def normalise(result: dict):
    pc = ENHARMONIC.get(result['key'], result['key'])
    return pc, result['mode']


def semitone_dist(a, b):
    if a not in PC_NAMES or b not in PC_NAMES:
        return -1
    d = abs(PC_NAMES.index(a) - PC_NAMES.index(b))
    return min(d, 12 - d)


# ---------------------------------------------------------------------------
# Evaluation engine
# ---------------------------------------------------------------------------
def evaluate_method(name, detect_fn, files, share_dir, verbose=False):
    total = key_correct = full_correct = mode_correct = skipped = errors = 0
    confidences = []
    interval_errors = defaultdict(int)
    mode_conf = defaultdict(int)

    for fname in files:
        stem = os.path.splitext(fname)[0]
        gt_pc, gt_mode = parse_filename_key(stem)
        if gt_pc is None:
            skipped += 1
            continue

        midi_path = os.path.join(share_dir, fname)
        try:
            result = detect_fn(midi_path)
        except Exception as e:
            errors += 1
            if verbose:
                print(f"  ERROR [{name}]: {fname} — {e}")
            continue

        pred_pc, pred_mode = normalise(result)
        conf = result.get('confidence', 0.0)
        confidences.append(conf)

        total += 1
        pc_ok = pred_pc == gt_pc
        mode_ok = pred_mode == gt_mode
        key_correct += int(pc_ok)
        mode_correct += int(mode_ok)
        full_correct += int(pc_ok and mode_ok)
        mode_conf[(gt_mode, pred_mode)] += 1

        if not pc_ok:
            interval_errors[semitone_dist(gt_pc, pred_pc)] += 1

        if verbose and not (pc_ok and mode_ok):
            status = 'PC_OK' if pc_ok else 'WRONG'
            print(f"  [{status}] {fname}")
            print(f"    GT: {gt_pc} {gt_mode}  |  Pred: {pred_pc} {pred_mode}  (conf={conf:.3f})")

    return {
        'name': name, 'total': total, 'skipped': skipped, 'errors': errors,
        'key_correct': key_correct, 'mode_correct': mode_correct, 'full_correct': full_correct,
        'confidences': confidences,
        'interval_errors': interval_errors,
        'mode_conf': mode_conf,
    }


def print_report(r):
    n = r['total']
    print(f"\n{'='*60}")
    print(f"METHOD: {r['name']}")
    print(f"{'='*60}")
    print(f"  Files evaluated         : {n}")
    print(f"  Skipped (bad GT label)  : {r['skipped']}")
    print(f"  Errors (parse failures) : {r['errors']}")
    if n:
        print(f"\n  Pitch-class accuracy    : {r['key_correct']}/{n} = {100*r['key_correct']/n:.1f}%")
        print(f"  Mode accuracy           : {r['mode_correct']}/{n} = {100*r['mode_correct']/n:.1f}%")
        print(f"  Full key+mode accuracy  : {r['full_correct']}/{n} = {100*r['full_correct']/n:.1f}%")
        if r['confidences']:
            print(f"  Mean confidence         : {np.mean(r['confidences']):.3f}")
        mc = r['mode_conf']
        print(f"\n  Mode confusion (rows=GT, cols=Pred):")
        for gt_m in ['major', 'minor']:
            for pr_m in ['major', 'minor']:
                print(f"    GT={gt_m:5s}  Pred={pr_m:5s} : {mc.get((gt_m, pr_m), 0)}")
        ie = r['interval_errors']
        if ie:
            print(f"\n  Wrong-key semitone distances:")
            for dist, cnt in sorted(ie.items(), key=lambda x: -x[1])[:8]:
                print(f"    {dist:2d} semitones : {cnt}")
    print('='*60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
SHARE_DIR = '/data/home/fundwotsai/MIDI-SAG/lyrics2melody/share'

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default=SHARE_DIR)
    p.add_argument('--method',
                   choices=['ks', 'demo_utils', 'improved',
                            'partitura_ks', 'partitura_kp', 'partitura_cmbs',
                            'music21_ks', 'music21_aarden', 'all'],
                   default='improved')
    p.add_argument('--verbose', action='store_true', help='Print each wrong prediction')
    p.add_argument('--max', type=int, default=None, help='Limit to first N files')
    args = p.parse_args()

    files = sorted(f for f in os.listdir(args.dir) if f.endswith('.mid'))
    if args.max:
        files = files[:args.max]

    ALL_METHODS = [
        ('ks',             'KS (demo_SOME.py)',              ks_key_detect),
        ('demo_utils',     'Scale-fit (demo_utils.py)',      demo_utils_key_detect),
        ('improved',       'Improved KS+leading-tone',       improved_key_detect),
        ('partitura_ks',   'Partitura KK',                   partitura_ks_detect),
        ('partitura_kp',   'Partitura Kostka-Payne',         partitura_kp_detect),
        ('partitura_cmbs', 'Partitura Bellman-Budge',        partitura_cmbs_detect),
        ('music21_ks',     'music21 KS',                     music21_ks_detect),
        ('music21_aarden', 'music21 Aarden-Essen',           music21_aarden_detect),
    ]
    methods = [(name, fn) for key, name, fn in ALL_METHODS
               if args.method == 'all' or args.method == key]

    results = []
    for name, fn in methods:
        print(f"\nRunning: {name} on {len(files)} files ...")
        r = evaluate_method(name, fn, files, args.dir, verbose=args.verbose)
        print_report(r)
        results.append(r)

    if len(results) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        for r in results:
            nn = r['total']
            print(f"  {r['name']:<38} PC={100*r['key_correct']/nn:.1f}%  "
                  f"Mode={100*r['mode_correct']/nn:.1f}%  "
                  f"Full={100*r['full_correct']/nn:.1f}%")
        print('='*60)
