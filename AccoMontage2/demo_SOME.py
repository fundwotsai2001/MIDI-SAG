import argparse
import chorderator as cdt
import os
import shutil
import tempfile
import mido
import json
import numpy as np
from datetime import datetime
from acc2btc import acc2btc
from miditok import REMI, TokenizerConfig
from symusic import Score
from pretty_midi import PrettyMIDI, Instrument, Note
config = TokenizerConfig(num_velocities=16, use_chords=False, use_programs=False)
tokenizer = REMI(config)
from demo_utils import (get_key,
                       get_chord_analysis,
                       get_advanced_chord_analysis,
                       get_detailed_key_analysis,
                       get_key_for_cdt,
                       get_mode_for_cdt,
                       get_auto_config,
                       fill_empty_bars_with_chords,
                       export_chords_txt,
                       export_chords_txt_chorder,
                       sync_output_tempo_with_input,
                       preprocess_melody,
                       align_chord_gen_tpq,
                       load_beat_times,
                       estimate_tempo_from_beats,
                       warp_midi_to_beats,
                       detect_downbeat_phase,
                       requantize_chord_gen_melody,
                       fill_none_chords_in_txt,
                       align_chord_track_to_bar,
                       scale_midi_ticks,
                       cover_pickup_melody_with_chord,
                       fill_internal_empty_bars
                       )

def melody_key_detect(midi_path):
    """
    Key detection for monophonic melody MIDI.
    Returns {'key': str, 'mode': str, 'confidence': float}.

    Uses Bellman-Budge (Temperley 2007) pitch-class profiles with
    Krumhansl-Schmuckler Pearson correlation scoring over all 24 keys.
    Bellman-Budge weights the tonic (10) and diatonic scale degrees far
    more sharply than the standard KS profiles (tonic weight 6.35),
    which greatly reduces relative-key confusion on pentatonic-heavy
    Chinese pop melodies — benchmarked at 77.7% full-key accuracy vs
    46.8% for standard KS on 988 labelled files.
    """
    import numpy as np

    # Bellman-Budge major/minor profiles (C-rooted, unnormalized)
    # Source: Temperley (2007) via partitura CMBS profiles
    BB_MAJOR = np.array([10., 4., 7., 4., 9., 8., 4., 9., 4., 7., 3., 8.])
    BB_MINOR = np.array([10., 4., 7., 9., 4., 8., 4., 9., 7., 4., 3., 8.])

    NOTE_NAMES  = ['C', 'C#', 'D', 'D#', 'E', 'F',
                   'F#', 'G', 'G#', 'A', 'A#', 'B']
    SHARP_TO_FLAT = {'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb', 'G#': 'Ab', 'A#': 'Bb'}

    # Build duration-weighted pitch-class histogram from all tracks
    mid = mido.MidiFile(midi_path)
    tpb = mid.ticks_per_beat
    tempo = 500000
    pc_dur = np.zeros(12)
    last_pitch = None

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
                last_pitch = msg.note
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                k = (msg.channel, msg.note)
                if k in active:
                    pc_dur[msg.note % 12] += t - active.pop(k)

    if pc_dur.sum() == 0:
        return {'key': 'C', 'mode': 'major', 'confidence': 0.0}

    v = pc_dur - pc_dur.mean()

    # Pearson correlation against all 24 Bellman-Budge key profiles
    best_r, best_root, best_mode = -2.0, 0, 'major'
    for root in range(12):
        maj = np.roll(BB_MAJOR, root); maj -= maj.mean()
        r = np.dot(v, maj) / (np.linalg.norm(v) * np.linalg.norm(maj) + 1e-9)
        if r > best_r: best_r, best_root, best_mode = r, root, 'major'

        min_ = np.roll(BB_MINOR, root); min_ -= min_.mean()
        r = np.dot(v, min_) / (np.linalg.norm(v) * np.linalg.norm(min_) + 1e-9)
        if r > best_r: best_r, best_root, best_mode = r, root, 'minor'

    best_key = SHARP_TO_FLAT.get(NOTE_NAMES[best_root], NOTE_NAMES[best_root])
    return {'key': best_key, 'mode': best_mode, 'confidence': float(best_r)}


# Keys accepted by --key. "auto" runs MIDI-metadata + Bellman-Budge detection;
# any other value is interpreted as a manual override (append "m" for minor).
AVAILABLE_KEYS = [
    'auto',
    'C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'F', 'F#', 'Gb',
    'G', 'G#', 'Ab', 'A', 'A#', 'Bb', 'B',
    'Cm', 'C#m', 'Dbm', 'Dm', 'D#m', 'Ebm', 'Em', 'Fm', 'F#m', 'Gbm',
    'Gm', 'G#m', 'Abm', 'Am', 'A#m', 'Bbm', 'Bm',
]


def parse_downbeat_phase(value):
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"none", "auto", ""}:
        return None
    return int(value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate chord progression from a single vocal MIDI file.')
    parser.add_argument('--midi_path', type=str, required=True, help='Path to input vocal MIDI file')
    parser.add_argument('--beat_file', type=str, default=None,
                        help='Path to beat times txt file. If omitted, beats are derived from the MIDI tempo.')
    parser.add_argument('--beat_file_detected', type=str, default=None,
                        help='Optional path to the detected-only beat times txt file (before gap-filling). '
                             'When provided together with --downbeat_phase, the phase indexes into this '
                             'file to pick the anchor downbeat time; downbeats are then bar_duration '
                             'multiples of that anchor.')
    parser.add_argument('--output_dir', type=str, default='output_SOME', help='Base output directory')
    parser.add_argument('--beat_subdivision', type=int, default=1,
                        help='Beat subdivision: 1=quarter, 2=8th, 4=16th notes (default: 1)')
    parser.add_argument('--chord_style', type=str, default='POP_STANDARD',
                        choices=['POP_STANDARD', 'POP_COMPLEX', 'DARK', 'RANDB', 'NOCONSTRAINT'],
                        help='Output chord style (default: POP_STANDARD)')
    parser.add_argument('--chords_per_bar', type=int, default=1, choices=[1, 2],
                        help='Chord events per bar. 2 splits each bar into two half-bar chord '
                             'events; the second half uses the next bar\'s chord (anticipation). '
                             'Default: 1')
    parser.add_argument('--downbeat_phase', type=parse_downbeat_phase, default=None,
                        help='Beat index to treat as bar 1 beat 1. When --beat_file_detected is given, '
                             'this is the index into the detected-only beat file; otherwise it is a '
                             'phase in [0, 4*beat_subdivision-1] into --beat_file. '
                             'Use None or auto to detect automatically.')
    parser.add_argument('--key', type=str, default='auto', choices=AVAILABLE_KEYS,
                        metavar='KEY',
                        help='Musical key of the melody. Accepts major roots (C, C#, Db, D, ..., B) '
                             'and their minor counterparts with an "m" suffix (Cm, C#m, ..., Bm). '
                             'Default "auto" reads MIDI key_signature metadata and falls back to '
                             'the Bellman-Budge heuristic over the melody pitch histogram.')
    args = parser.parse_args()

    input_melody_path = args.midi_path
    BEAT_SUBDIVISION = args.beat_subdivision
    max_downbeat_phase = 4 * BEAT_SUBDIVISION - 1
    if (args.downbeat_phase is not None
            and args.beat_file_detected is None
            and not 0 <= args.downbeat_phase <= max_downbeat_phase):
        raise ValueError(
            f"--downbeat_phase must be between 0 and {max_downbeat_phase} "
            f"for beat_subdivision={BEAT_SUBDIVISION}; got {args.downbeat_phase}"
        )
    if args.downbeat_phase is not None and args.downbeat_phase < 0:
        raise ValueError(
            f"--downbeat_phase must be non-negative; got {args.downbeat_phase}"
        )

    # Derive song name from filename stem
    song_name = os.path.splitext(os.path.basename(input_melody_path))[0]
    demo_name = song_name.replace(" ", "_").replace("/", "_")

    # Create output directory structure
    output_base_dir = args.output_dir
    chord_gen_filled_dir = os.path.join(output_base_dir, "chord_gen_filled_empty")
    btc_txt_dir = os.path.join(output_base_dir, "btc_txt")

    os.makedirs(chord_gen_filled_dir, exist_ok=True)
    os.makedirs(btc_txt_dir, exist_ok=True)

    # Intermediate artifacts live in a tempdir and are removed on exit.
    _tmp_root = tempfile.mkdtemp(prefix="demo_SOME_")
    processed_melody_dir = os.path.join(_tmp_root, "processed_melody")
    chord_gen_dir = os.path.join(_tmp_root, "chord_gen")
    chord_txt_with_none_dir = os.path.join(_tmp_root, "chord_txt_with_None")
    chord_txt_dir = os.path.join(_tmp_root, "chord_txt")

    os.makedirs(processed_melody_dir, exist_ok=True)
    os.makedirs(chord_gen_dir, exist_ok=True)
    os.makedirs(chord_txt_with_none_dir, exist_ok=True)
    os.makedirs(chord_txt_dir, exist_ok=True)

    print(f"\n=== Processing: {song_name} ===")
    print(f"Output will be saved to: {output_base_dir}/")

    processed_melody_path = os.path.join(processed_melody_dir, f"{demo_name}.mid")

    beat_file_path = args.beat_file
    print(f"beat_file_path: {beat_file_path}")
    try:
        # Step 1: strip to a single melody track
        score = Score(input_melody_path, ttype="tick")
        for track in score.tracks:
            if track.notes:
                score.tracks = [track]
                break
        score.dump_midi(processed_melody_path)

        if beat_file_path is not None:
            if not os.path.exists(beat_file_path):
                raise FileNotFoundError(f"Beat times file not found: {beat_file_path}")

            beat_times = load_beat_times(beat_file_path)
            tempo_info = estimate_tempo_from_beats(beat_times, beat_subdivision=BEAT_SUBDIVISION)
            estimated_bpm = tempo_info['tempo']
            print(f"Tempo from beats: {estimated_bpm:.1f} BPM (bar duration: {tempo_info['bar_duration']:.3f}s)")

            # Step 2: choose which beat is bar 1 beat 1
            downbeat_stride = 4 * BEAT_SUBDIVISION
            if args.downbeat_phase is None:
                downbeat_phase = detect_downbeat_phase(
                    beat_times,
                    processed_melody_path,
                    beats_per_bar=4,
                    beat_subdivision=BEAT_SUBDIVISION,
                )
            elif args.beat_file_detected is not None:
                # The detected-only beat file just picks a concrete downbeat
                # time; we convert that into a phase within a bar so the full
                # beat grid (including pickup beats before the anchor) is
                # still used for warping and downbeat emission.
                if not os.path.exists(args.beat_file_detected):
                    raise FileNotFoundError(
                        f"Detected beat times file not found: {args.beat_file_detected}"
                    )
                detected_beats = load_beat_times(args.beat_file_detected)
                if args.downbeat_phase >= len(detected_beats):
                    raise ValueError(
                        f"--downbeat_phase={args.downbeat_phase} is out of range for "
                        f"detected beats of length {len(detected_beats)}"
                    )
                anchor_time = float(detected_beats[args.downbeat_phase])
                anchor_idx = int(np.argmin(np.abs(np.asarray(beat_times) - anchor_time)))
                downbeat_phase = anchor_idx % downbeat_stride
                print(
                    f"  [phase] anchor = detected[{args.downbeat_phase}] = {anchor_time:.3f}s "
                    f"→ beat_times idx {anchor_idx} (t={beat_times[anchor_idx]:.3f}s), "
                    f"phase within bar = {downbeat_phase}"
                )
            else:
                downbeat_phase = args.downbeat_phase
                print(f"  [phase] using user-provided downbeat phase={downbeat_phase}")

            # Emit downbeats for the whole beat grid on the bar cycle anchored
            # at downbeat_phase (x can be positive or negative).
            downbeat_times = beat_times[downbeat_phase::downbeat_stride]
            downbeat_out_path = os.path.join(
                os.path.dirname(beat_file_path),
                f"{os.path.splitext(os.path.basename(beat_file_path))[0].replace('_beat_times', '')}_downbeat_times.txt"
            )
            with open(downbeat_out_path, 'w') as f:
                f.write(f"# Downbeat times in seconds (phase={downbeat_phase}, stride={downbeat_stride})\n")
                for t in downbeat_times:
                    f.write(f"{t:.6f}\n")
            print(f"  [phase] saved {len(downbeat_times)} downbeat times → {downbeat_out_path}")

            # Step 3: warp note positions onto the beat grid
            beat_times_aligned = beat_times[downbeat_phase:]
            warp_midi_to_beats(
                processed_melody_path, processed_melody_path,
                beat_times_aligned,
                beats_per_bar=4,
                quantize_to_grid=True,
                grid_resolution=16
            )
        else:
            # Derive beats directly from the MIDI tempo (already on-grid, no warping needed)
            import mido as _mido
            _mid = _mido.MidiFile(processed_melody_path)
            _tempo = next(
                (msg.tempo for track in _mid.tracks for msg in track if msg.type == 'set_tempo'),
                500000
            )
            estimated_bpm = 60_000_000 / _tempo
            beat_interval = _tempo / 1_000_000  # seconds per beat
            total_time = _mid.length
            beat_times = [round(i * beat_interval, 6) for i in range(int(total_time / beat_interval) + 2)]
            beat_times_aligned = beat_times  # MIDI tick 0 = bar 1 beat 1, no phase offset
            print(f"Tempo from MIDI: {estimated_bpm:.1f} BPM (no beat file — skipping warp step)")

        # For chords_per_bar == 2, feed chorderator a tick-scaled melody so each
        # real half-bar appears as a full bar. Chorderator then picks a distinct
        # chord per half-bar from that half-bar's melody content. The generated
        # chord MIDI is descaled back to real time below.
        if args.chords_per_bar == 2:
            scaled_melody_path = os.path.join(
                processed_melody_dir, f"{demo_name}_x{args.chords_per_bar}.mid"
            )
            scale_midi_ticks(processed_melody_path, scaled_melody_path,
                             args.chords_per_bar)
            cdt_melody_path = scaled_melody_path
        else:
            cdt_melody_path = processed_melody_path

        # Process the MIDI file
        cdt.set_melody(cdt_melody_path)
        print(f"cdt_melody_path: {cdt_melody_path}")
        midi_obj = Score(cdt_melody_path)
        tokens = tokenizer(midi_obj)
        if len(tokens) == 1:
            tokens = tokens[0]

        # Key analysis: honor --key when specified; otherwise prefer MIDI
        # key_signature metadata and fall back to the Bellman-Budge heuristic.
        if args.key != 'auto':
            _is_minor = args.key.endswith('m')
            key_analysis = {
                'key': args.key[:-1] if _is_minor else args.key,
                'mode': 'minor' if _is_minor else 'major',
                'confidence': 1.0,
            }
            print(f"Using user-specified key: {key_analysis['key']} {key_analysis['mode']}")
        else:
            _midi_key_sig = None
            for _track in mido.MidiFile(input_melody_path).tracks:
                for _msg in _track:
                    if _msg.type == 'key_signature':
                        _midi_key_sig = _msg.key  # e.g. 'Db', 'Am', 'F#'
                        break
                if _midi_key_sig:
                    break

            if _midi_key_sig is not None:
                _is_minor = _midi_key_sig.endswith('m')
                key_analysis = {
                    'key': _midi_key_sig[:-1] if _is_minor else _midi_key_sig,
                    'mode': 'minor' if _is_minor else 'major',
                    'confidence': 1.0,
                }
                print(f"Detected key: {key_analysis['key']} {key_analysis['mode']} (from MIDI key signature)")
            else:
                key_analysis = melody_key_detect(input_melody_path)
                print(f"Detected key: {key_analysis['key']} {key_analysis['mode']} (confidence: {key_analysis['confidence']:.2f}, Bellman-Budge)")

        cdt_key_attr = get_key_for_cdt(tokens.tokens, key_analysis)
        cdt_mode_attr = get_mode_for_cdt(tokens.tokens, key_analysis)

        # Auto-configure (use MIDI file for accurate bar counting).
        # When chords_per_bar == 2, the scaled melody naturally yields double
        # the bar count, and segmentation (A8 → A16, etc.) doubles accordingly.
        auto_config = get_auto_config(tokens.tokens, midi_path=cdt_melody_path)
        print(f"auto_config: {auto_config}")

        # Chorderator sees each real half-bar as a full bar, so its internal
        # tempo is double the real song tempo.
        tempo = estimated_bpm * args.chords_per_bar
        print(f"tempo for CDT: {tempo:.1f} BPM (chords_per_bar={args.chords_per_bar})")

        # Set parameters
        cdt_key_value = getattr(cdt.Key, cdt_key_attr)
        cdt_mode_value = getattr(cdt.Mode, cdt_mode_attr)
        cdt.set_meta(tonic=cdt_key_value, mode=cdt_mode_value, tempo=tempo)
        cdt.set_note_shift(auto_config['note_shift'])
        cdt.set_segmentation(auto_config['segmentation'])
        print("args.chord_style", args.chord_style)
        cdt.set_output_style(getattr(cdt.Style, args.chord_style))

        # Generate chord progression
        chord_gen_output = os.path.join(chord_gen_dir, f"{demo_name}_chord_gen.mid")
        chord_gen = cdt.generate_save(output_dir=chord_gen_dir,
                                      chord_output_name=f"{demo_name}_chord_gen.mid",
                                      task='chord',
                                      log=False)

        # Descale chord MIDI back to real time when we fed chorderator a scaled
        # melody. After this, one chord-bar = one real half-bar, giving
        # `chords_per_bar` distinct chords per real bar.
        if args.chords_per_bar == 2:
            scale_midi_ticks(chord_gen_output, chord_gen_output,
                             1.0 / args.chords_per_bar)

        # Align chord_gen TPQ with original melody TPQ
        align_chord_gen_tpq(processed_melody_path, chord_gen_output)

        # Re-quantize melody track to fix chorderator's rounding errors
        requantize_chord_gen_melody(chord_gen_output)

        # Fill empty bars
        # auto_config was computed from the (possibly scaled) cdt_melody_path,
        # so empty_bars is in scaled-bar units; convert back to real bars.
        empty_bars = auto_config['analysis']['empty_bars'] // args.chords_per_bar
        filled_output = os.path.join(chord_gen_filled_dir, f"{demo_name}_chord_gen_filled_empty_bars.mid")
        fill_empty_bars_with_chords(
            processed_melody_path,
            chord_gen_output,
            empty_bars,
            filled_output
        )

        # Forward-fill any mid-song bar where chorderator emitted no chord.
        fill_internal_empty_bars(filled_output)

        # Snap chord-event boundaries to MIDI grid (DAW-friendly).
        # For chords_per_bar==2, snap to half-bar so the post-fill chord
        # sequence stays half-bar aligned rather than being forced onto bar
        # boundaries.
        chord_resolution_beats = 4 // args.chords_per_bar
        align_chord_track_to_bar(filled_output,
                                 chord_resolution_beats=chord_resolution_beats)

        # Export chord text with None values
        txt_with_none = os.path.join(chord_txt_with_none_dir, f"{demo_name}_chord_gen.txt")
        export_chords_txt_chorder(filled_output, txt_with_none, beat_times=beat_times_aligned,
                                  beat_subdivision=BEAT_SUBDIVISION)

        # Fill None chords
        txt_file = os.path.join(chord_txt_dir, f"{demo_name}_chord_gen.txt")
        fill_none_chords_in_txt(txt_with_none, txt_file)

        # Convert to BTC format
        btc_file = os.path.join(btc_txt_dir, f"{demo_name}_chord_gen.txt")
        acc2btc(txt_file, btc_file)

        # Extend the first chord back over the melody pickup in the MIDI too.
        # Text/BTC leading silence is handled separately in fill_none_chords_in_txt.
        cover_pickup_melody_with_chord(filled_output)

        print(f"✓ Successfully processed: {song_name}")
        print(f"  Key: {key_analysis['key']} {key_analysis['mode']} (confidence: {key_analysis['confidence']:.2f})")
        print(f"  Tempo: {estimated_bpm:.1f} BPM (source: {'beat file' if beat_file_path else 'MIDI tempo'})")
        print(f"  BTC Text:   {btc_file}")
        print(f"  Filled MIDI: {filled_output}")

    except Exception as e:
        print(f"✗ Failed to process: {song_name}")
        print(f"  Error: {str(e)}")
        raise
    finally:
        shutil.rmtree(_tmp_root, ignore_errors=True)
