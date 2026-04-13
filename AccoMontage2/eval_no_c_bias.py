"""
Compare demo_utils get_detailed_key_analysis (original) vs
a patched version with the C-major tie-breaker removed,
on both share and generated_midi datasets.
"""
import os, sys, glob, warnings, importlib
import numpy as np
from collections import defaultdict

warnings.filterwarnings('ignore')
sys.path.insert(0, '/data/home/fundwotsai/MIDI-SAG/AccoMontage2')

SHARE_DIR     = '/data/home/fundwotsai/MIDI-SAG/lyrics2melody/share'
GENERATED_DIR = '/data/home/fundwotsai/MIDI-SAG/lyrics2melody/generated_midi'

# ---------------------------------------------------------------------------
# Normalisation helpers (same as eval_all_methods.py)
# ---------------------------------------------------------------------------
ENHARMONIC = {
    'C':'C','B#':'C','Db':'Db','C#':'Db','D':'D','Eb':'Eb','D#':'Eb',
    'E':'E','Fb':'E','F':'F','E#':'F','Gb':'Gb','F#':'Gb','G':'G',
    'Ab':'Ab','G#':'Ab','A':'A','Bb':'Bb','A#':'Bb','B':'B','Cb':'B',
    'bB':'Bb','bE':'Eb','bA':'Ab','bD':'Db','bG':'Gb',
    '#F':'Gb','#C':'Db','#G':'Ab','#D':'Eb','#A':'Bb',
}
PC_NAMES = ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']

def normalise(result):
    return ENHARMONIC.get(result['key'], result['key']), result['mode']

def parse_filename_key(stem):
    parts = stem.split('-')
    if len(parts) < 2: return None, None
    raw = parts[-1].strip()
    mode, raw_pc = ('minor', raw[:-1]) if raw.endswith('m') else ('major', raw)
    pc = ENHARMONIC.get(raw_pc)
    return (pc, mode) if pc else (None, None)

# ---------------------------------------------------------------------------
# Tokenizer (shared)
# ---------------------------------------------------------------------------
from miditok import REMI, TokenizerConfig
from symusic import Score as SymScore
_tok = REMI(TokenizerConfig(num_velocities=16, use_chords=False, use_programs=False))

def tokenize(midi_path):
    tokens = _tok(SymScore(midi_path))
    if len(tokens) == 1: tokens = tokens[0]
    return tokens.tokens

# ---------------------------------------------------------------------------
# Original demo_utils detect
# ---------------------------------------------------------------------------
def original_detect(midi_path):
    du = importlib.import_module('demo_utils')
    r = du.get_detailed_key_analysis(tokenize(midi_path))
    return {'key': r['key'], 'mode': r['mode'], 'confidence': r['confidence']}

# ---------------------------------------------------------------------------
# Patched: remove C-major tie-breaker + remove C-first dict ordering bias
# ---------------------------------------------------------------------------
pitch_to_key = {0:'C',1:'C#',2:'D',3:'D#',4:'E',5:'F',
                6:'F#',7:'G',8:'G#',9:'A',10:'A#',11:'B'}

def _no_c_bias_detect(tokens):
    # Same scoring as get_detailed_key_analysis
    major_keys = ['C','G','D','A','E','B','F#','C#','G#','D#','A#','F']
    minor_keys = ['A','E','B','F#','C#','G#','D#','A#','F','C','G','D']

    pitch_classes = []
    for token in tokens:
        if token.startswith('Pitch'):
            pitch_classes.append(int(token.split('_')[1]) % 12)
    if not pitch_classes:
        return {'key': 'C', 'mode': 'major', 'confidence': 0.0}

    pc_count = {}
    for pc in pitch_classes:
        pc_count[pc] = pc_count.get(pc, 0) + 1
    total_notes = len(pitch_classes)

    key_analysis = {}
    for key in major_keys:
        key_pc = list(pitch_to_key.keys())[list(pitch_to_key.values()).index(key)]
        scale = [(key_pc + i) % 12 for i in [0,2,4,5,7,9,11]]
        scale_notes = sum(1 for pc in pitch_classes if pc in scale)
        confidence = scale_notes / total_notes
        tonic_bonus = 0.1 if key_pc in pitch_classes else 0
        key_analysis[f"{key}_major"] = {
            'key': key, 'mode': 'major',
            'confidence': min(confidence + tonic_bonus, 1.0)
        }
    for key in minor_keys:
        key_pc = list(pitch_to_key.keys())[list(pitch_to_key.values()).index(key)]
        scale = [(key_pc + i) % 12 for i in [0,2,3,5,7,8,10]]
        scale_notes = sum(1 for pc in pitch_classes if pc in scale)
        confidence = scale_notes / total_notes
        tonic_bonus  = 0.1  if key_pc in pitch_classes else 0
        m3_bonus     = 0.05 if (key_pc+3)%12 in pitch_classes else 0
        m6_bonus     = 0.05 if (key_pc+8)%12 in pitch_classes else 0
        key_analysis[f"{key}_minor"] = {
            'key': key, 'mode': 'minor',
            'confidence': min(confidence + tonic_bonus + m3_bonus + m6_bonus, 1.0)
        }

    best_confidence = max(v['confidence'] for v in key_analysis.values())
    tied = [k for k, v in key_analysis.items()
            if abs(v['confidence'] - best_confidence) < 0.001]

    # --- NO C-major tie-breaker ---
    # Among ties: prefer same-key major over minor; then major over minor generally.
    # When multiple different major keys are tied, pick by highest tonic note count.
    if len(tied) > 1:
        major_tied = [k for k in tied if k.endswith('_major')]
        minor_tied = [k for k in tied if k.endswith('_minor')]

        # Prefer matching major over its own minor (e.g. C_major over C_minor)
        chosen = None
        for mk in minor_tied:
            corresp = mk.replace('_minor', '_major')
            if corresp in major_tied:
                chosen = corresp; break

        if chosen is None and major_tied:
            # Multiple different major keys tied — pick by tonic note frequency
            def tonic_count(k):
                kname = k.replace('_major','').replace('_minor','')
                kpc = list(pitch_to_key.keys())[list(pitch_to_key.values()).index(kname)]
                return pc_count.get(kpc, 0)
            chosen = max(major_tied, key=tonic_count)

        if chosen is None and minor_tied:
            chosen = max(minor_tied, key=lambda k: pc_count.get(
                list(pitch_to_key.keys())[list(pitch_to_key.values()).index(
                    k.replace('_minor',''))], 0))

        best_key_name = chosen or tied[0]
    else:
        best_key_name = tied[0]

    best = key_analysis[best_key_name]
    return {'key': best['key'], 'mode': best['mode'], 'confidence': best['confidence']}

def patched_detect(midi_path):
    r = _no_c_bias_detect(tokenize(midi_path))
    return r

# ---------------------------------------------------------------------------
# Evaluation engine
# ---------------------------------------------------------------------------
def evaluate(name, detect_fn, midi_paths, gt_fn):
    total = key_ok = mode_ok = full_ok = errors = skipped = 0
    wrong = defaultdict(int)

    for path in midi_paths:
        gt_pc, gt_mode = gt_fn(path)
        if gt_pc is None: skipped += 1; continue
        total += 1
        try:
            result = detect_fn(path)
        except Exception as e:
            errors += 1; total -= 1
            print(f"  ERROR {os.path.basename(path)}: {e}"); continue
        pred_pc, pred_mode = normalise(result)
        pc_ok = pred_pc == gt_pc
        mo_ok = pred_mode == gt_mode
        key_ok  += int(pc_ok)
        mode_ok += int(mo_ok)
        full_ok += int(pc_ok and mo_ok)
        if not pc_ok:
            wrong[f"{pred_pc} {pred_mode}"] += 1

    n = total
    print(f"  {'Files':6} {n}  Skipped {skipped}  Errors {errors}")
    if n:
        print(f"  Pitch-class  : {key_ok}/{n} = {100*key_ok/n:.1f}%")
        print(f"  Mode         : {mode_ok}/{n} = {100*mode_ok/n:.1f}%")
        print(f"  Full key+mode: {full_ok}/{n} = {100*full_ok/n:.1f}%")
    if wrong:
        print(f"  Top wrong predictions:")
        for k, c in sorted(wrong.items(), key=lambda x:-x[1])[:6]:
            print(f"    {k:<15}: {c}")
    return full_ok, n

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    share_files = sorted(glob.glob(os.path.join(SHARE_DIR, '*.mid')))
    gen_files   = sorted(glob.glob(os.path.join(GENERATED_DIR, 'lyrics_*/sample01.mid')))

    def share_gt(p): return parse_filename_key(os.path.splitext(os.path.basename(p))[0])
    def gen_gt(p):   return 'C', 'major'

    methods = [
        ('Original (with C-major tie-breaker)', original_detect),
        ('Patched  (no C-major bias)',          patched_detect),
    ]

    for dataset_name, files, gt_fn in [
        ('share (1000 files, varied keys)', share_files, share_gt),
        ('generated_midi (200 files, C major)', gen_files, gen_gt),
    ]:
        print(f"\n{'='*58}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*58}")
        summary = []
        for name, fn in methods:
            print(f"\n  [{name}]")
            full_ok, n = evaluate(name, fn, files, gt_fn)
            summary.append((name, full_ok, n))
        print(f"\n  -- Summary --")
        for name, full_ok, n in summary:
            pct = f"{100*full_ok/n:.1f}%" if n else "N/A"
            print(f"  {name:<45} {full_ok}/{n} = {pct}")
