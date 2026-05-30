from typing import Optional
from miditok import REMI, TokenizerConfig
from symusic import Score
config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True, use_tempos=True)
tokenizer = REMI(config)
import copy
import mido
import os

from chorder import Dechorder
import numpy as np


def load_beat_times(beat_file_path):
    """
    Load beat times from a text file (SingNet format).
    
    Args:
        beat_file_path: Path to beat_times.txt file
    
    Returns:
        numpy array of beat times in seconds
    """
    beat_times = []
    with open(beat_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    beat_times.append(float(line))
                except ValueError:
                    continue
    return np.array(beat_times)


def estimate_tempo_from_beats(beat_times, beats_per_bar=4, prefer_tempo_range=(60, 150), beat_subdivision=1):
    """
    Estimate tempo from beat times with octave correction.
    
    Args:
        beat_times: Array of beat times in seconds
        beats_per_bar: Number of beats per bar (default 4 for 4/4)
        prefer_tempo_range: Preferred tempo range (min, max) for octave correction
        beat_subdivision: What subdivision the beat_times represent (default 1 = quarter notes)
                         - 1 = quarter notes (beats)
                         - 2 = 8th notes
                         - 4 = 16th notes
                         If beat_times are 8th notes, set to 2 to get correct quarter note tempo.
    
    Returns:
        dict: {
            'tempo': float (BPM, in quarter notes per minute),
            'mean_ibi': float (mean inter-beat interval in seconds, for quarter notes),
            'bar_duration': float (beats_per_bar quarter notes in seconds),
            'std_ibi': float (standard deviation of IBIs),
            'beats_per_bar': int,
            'octave_corrected': bool,
            'beat_subdivision': int
        }
    """
    if len(beat_times) < 2:
        return {
            'tempo': 120.0, 
            'mean_ibi': 0.5, 
            'bar_duration': 2.0, 
            'std_ibi': 0.0,
            'beats_per_bar': beats_per_bar,
            'octave_corrected': False,
            'beat_subdivision': beat_subdivision
        }
    
    # Calculate inter-beat intervals
    ibis = np.diff(beat_times)

    # Robust IBI: drop outliers before averaging. Detected-only beat files
    # (vocal segments only, no gap-filling) contain large pseudo-IBIs across
    # instrumental breaks where no vocal beats were detected — e.g. a 25s gap
    # between two vocal sections. A naive np.mean is badly contaminated by
    # those gaps (BKS: mean=0.944s vs median=0.800s → 63.58 BPM vs 75 BPM).
    # Trim IBIs that deviate from the median by more than 50% and average the
    # rest; for continuous beat arrays (interpolated files, gt MIDI), nothing
    # gets trimmed and this is equivalent to the old mean-based estimate.
    median_ibi = float(np.median(ibis))
    inlier_mask = np.abs(ibis - median_ibi) <= 0.5 * median_ibi
    if inlier_mask.sum() >= max(3, len(ibis) // 4):
        raw_mean_ibi = float(np.mean(ibis[inlier_mask]))
    else:
        # Degenerate case (too few inliers) — fall back to median
        raw_mean_ibi = median_ibi
    std_ibi = float(np.std(ibis[inlier_mask])) if inlier_mask.any() else float(np.std(ibis))
    trimmed = int(len(ibis) - inlier_mask.sum())
    if trimmed > 0:
        print(f"  Trimmed {trimmed}/{len(ibis)} outlier IBIs "
              f"(median={median_ibi:.3f}s, robust mean={raw_mean_ibi:.3f}s)")

    # Adjust for subdivision: if beat_times are 8th notes, multiply IBI by 2 to get quarter note IBI
    mean_ibi = raw_mean_ibi * beat_subdivision
    tempo = 60.0 / mean_ibi
    
    if beat_subdivision > 1:
        print(f"  Beat subdivision: {beat_subdivision} (adjusting IBI from {raw_mean_ibi:.3f}s to {mean_ibi:.3f}s)")
    
    # Octave correction: if tempo is outside preferred range, try 2x or 0.5x
    octave_corrected = False
    min_tempo, max_tempo = prefer_tempo_range
    
    if tempo > max_tempo:
        # Tempo too fast, try halving it
        if tempo / 2 >= min_tempo:
            tempo = tempo / 2
            mean_ibi = mean_ibi * 2
            octave_corrected = True
            print(f"  Octave correction: tempo halved to {tempo:.1f} BPM")
    elif tempo < min_tempo:
        # Tempo too slow, try doubling it
        if tempo * 2 <= max_tempo:
            tempo = tempo * 2
            mean_ibi = mean_ibi / 2
            octave_corrected = True
            print(f"  Octave correction: tempo doubled to {tempo:.1f} BPM")
    
    bar_duration = mean_ibi * beats_per_bar
    
    return {
        'tempo': tempo,
        'mean_ibi': mean_ibi,
        'bar_duration': bar_duration,
        'std_ibi': std_ibi,
        'beats_per_bar': beats_per_bar,
        'octave_corrected': octave_corrected,
        'beat_subdivision': beat_subdivision
    }


def get_downbeat_times(beat_times, beats_per_bar=4, first_downbeat_idx=0):
    """
    Get downbeat (bar start) times from beat times.
    
    Args:
        beat_times: Array of beat times in seconds
        beats_per_bar: Number of beats per bar (default 4 for 4/4)
        first_downbeat_idx: Index of the first downbeat in beat_times
    
    Returns:
        Array of downbeat times in seconds
    """
    downbeats = []
    for i in range(first_downbeat_idx, len(beat_times), beats_per_bar):
        downbeats.append(beat_times[i])
    return np.array(downbeats)


def adjust_midi_to_beats(input_midi_path, output_midi_path, beat_times, beats_per_bar=4, beat_subdivision=1):
    """
    Adjust MIDI tempo based on beat times while PRESERVING absolute note timing (seconds).
    
    This function:
    1. Estimates correct tempo from beat times
    2. Rescales tick positions so notes stay at the SAME absolute seconds
    3. Updates tempo metadata to match the beat grid
    
    The notes will NOT be quantized to the beat grid - they stay at their
    original positions in seconds (as extracted from audio by SOME).
    
    Args:
        input_midi_path: Path to input MIDI file
        output_midi_path: Path to output MIDI file
        beat_times: Array of beat times in seconds (from SingNet)
        beats_per_bar: Number of beats per bar (default 4)
        beat_subdivision: What subdivision the beat_times represent (default 1 = quarter notes)
                         - 1 = quarter notes (beats)
                         - 2 = 8th notes
                         - 4 = 16th notes
    
    Returns:
        dict with tempo info
    """
    from symusic import Score, Note, Track, Tempo
    
    # Get tempo from beats
    tempo_info = estimate_tempo_from_beats(beat_times, beats_per_bar, beat_subdivision=beat_subdivision)
    new_tempo = tempo_info['tempo']
    
    print(f"=== Adjusting MIDI Tempo (preserving absolute seconds) ===")
    print(f"Estimated tempo from beats: {new_tempo:.1f} BPM")
    
    # Load original MIDI
    score = Score(input_midi_path, ttype="tick")
    original_tpq = score.ticks_per_quarter
    
    # Get original tempo
    original_tempo = 120.0
    if score.tempos:
        original_tempo = score.tempos[0].qpm
    
    print(f"Original tempo: {original_tempo:.1f} BPM")
    print(f"TPQ: {original_tpq}")
    
    # Calculate tick scaling factor to preserve absolute seconds
    # Original: seconds = ticks / (tpq * original_tempo / 60)
    # New: seconds = new_ticks / (tpq * new_tempo / 60)
    # To keep same seconds: new_ticks = ticks * (new_tempo / original_tempo)
    tempo_ratio = new_tempo / original_tempo
    
    print(f"Tempo ratio: {tempo_ratio:.3f}")
    
    # Create new score
    new_score = Score(ttype="tick")
    new_score.ticks_per_quarter = original_tpq
    new_score.tempos = [Tempo(time=0, qpm=new_tempo)]
    new_score.time_signatures = score.time_signatures.copy() if score.time_signatures else []
    
    # Scale all note positions to preserve absolute seconds
    for track in score.tracks:
        new_track = Track(name=track.name, program=track.program, is_drum=track.is_drum)
        for note in track.notes:
            new_start = int(round(note.time * tempo_ratio))
            new_duration = int(round(note.duration * tempo_ratio))
            if new_duration < 1:
                new_duration = 1
            
            new_note = Note(
                time=new_start,
                duration=new_duration,
                pitch=note.pitch,
                velocity=note.velocity
            )
            new_track.notes.append(new_note)
        new_track.notes.sort(key=lambda n: n.time)
        new_score.tracks.append(new_track)
    
    new_score.dump_midi(output_midi_path)
    
    print(f"Adjusted MIDI saved (absolute seconds preserved)")
    
    return {
        'original_tempo': original_tempo,
        'new_tempo': new_tempo,
        'tempo_ratio': tempo_ratio,
        'mean_ibi': tempo_info['mean_ibi'],
        'bar_duration': tempo_info['bar_duration']
    }


def warp_midi_to_beats(input_midi_path, output_midi_path, beat_times, beats_per_bar=4, 
                       quantize_to_grid=True, grid_resolution=16):
    """
    Warp MIDI notes to align with beat times while preserving relative timing within beats.
    
    This function performs beat-aware warping:
    1. For each note, find which beat interval it belongs to (in the original audio time)
    2. Calculate the note's relative position within that beat interval (0 to 1)
    3. Map the note to the corresponding position in the output MIDI beat grid
    
    This preserves the musical timing relative to beats while ensuring proper alignment.
    
    Args:
        input_midi_path: Path to input MIDI file
        output_midi_path: Path to output MIDI file  
        beat_times: Array of beat times in seconds (from SingNet or other beat tracker)
        beats_per_bar: Number of beats per bar (default 4 for 4/4)
        quantize_to_grid: If True, quantize notes to nearest grid position (default True)
        grid_resolution: Grid resolution in divisions per beat (default 16 = 1/16 note per beat)
    
    Returns:
        dict with tempo info and adjustment details
    """
    from symusic import Score, Note, Track, Tempo
    
    # Get tempo info from beats
    tempo_info = estimate_tempo_from_beats(beat_times, beats_per_bar)
    new_tempo = tempo_info['tempo']
    mean_ibi = tempo_info['mean_ibi']  # Mean inter-beat interval in seconds
    
    print(f"=== Beat-Aware MIDI Warping ===")
    print(f"Estimated tempo from beats: {new_tempo:.1f} BPM")
    print(f"Mean inter-beat interval: {mean_ibi:.3f}s")
    print(f"Bar duration: {tempo_info['bar_duration']:.3f}s")
    
    # Load the original MIDI
    score = Score(input_midi_path, ttype="tick")
    original_tpq = score.ticks_per_quarter
    
    # Get original tempo (needed to convert ticks to seconds)
    original_tempo = 120.0
    if score.tempos:
        original_tempo = score.tempos[0].qpm
    
    print(f"Original MIDI tempo: {original_tempo:.1f} BPM")
    print(f"Original TPQ: {original_tpq}")
    
    # Calculate seconds per tick for original MIDI
    original_secs_per_tick = 60.0 / (original_tempo * original_tpq)
    
    # Calculate ticks per beat for new MIDI
    # At new_tempo BPM, one beat = 60/new_tempo seconds
    # ticks_per_beat = tpq (since tpq = ticks per quarter note = ticks per beat in 4/4)
    ticks_per_beat = original_tpq
    ticks_per_grid = ticks_per_beat / grid_resolution
    
    print(f"Ticks per beat: {ticks_per_beat}")
    print(f"Grid resolution: 1/{grid_resolution} beat ({ticks_per_grid:.1f} ticks)")
    
    # Create beat interval lookup
    # beat_times[i] to beat_times[i+1] is beat interval i
    beat_intervals = []
    for i in range(len(beat_times) - 1):
        beat_intervals.append({
            'start': beat_times[i],
            'end': beat_times[i + 1],
            'duration': beat_times[i + 1] - beat_times[i]
        })
    # Add a final beat interval (estimate duration from mean IBI)
    if len(beat_times) > 0:
        beat_intervals.append({
            'start': beat_times[-1],
            'end': beat_times[-1] + mean_ibi,
            'duration': mean_ibi
        })
    
    def find_beat_and_position(time_seconds):
        """
        Find which beat interval a time belongs to and the relative position within it.
        
        Returns:
            (beat_index, relative_position) where relative_position is 0 to 1
        """
        # Handle time before first beat — continuous linear extrapolation so
        # multiple pickup notes keep their relative spacing instead of all
        # collapsing onto tick 0.
        if time_seconds < beat_times[0]:
            offset = beat_times[0] - time_seconds
            beats_before = offset / mean_ibi  # how many beats back, continuous
            beat_idx_f = -beats_before
            beat_idx = int(np.floor(beat_idx_f))
            rel_pos = beat_idx_f - beat_idx  # in [0, 1)
            return beat_idx, rel_pos
        
        # Binary search for the beat interval
        for i, interval in enumerate(beat_intervals):
            if interval['start'] <= time_seconds < interval['end']:
                rel_pos = (time_seconds - interval['start']) / interval['duration']
                return i, min(1.0, max(0.0, rel_pos))
        
        # Time is after the last beat - extrapolate
        time_after_last = time_seconds - beat_times[-1]
        beats_after = time_after_last / mean_ibi
        beat_idx = len(beat_times) - 1 + int(beats_after)
        rel_pos = (beats_after % 1.0)
        return beat_idx, rel_pos
    
    def quantize_to_nearest_grid(tick, grid_ticks):
        """Quantize tick to nearest grid position."""
        return int(round(tick / grid_ticks) * grid_ticks)
    
    # Create new score
    new_score = Score(ttype="tick")
    new_score.ticks_per_quarter = original_tpq
    new_score.tempos = [Tempo(time=0, qpm=new_tempo)]
    new_score.time_signatures = score.time_signatures.copy() if score.time_signatures else []
    
    # Process all notes
    notes_processed = 0
    notes_quantized = 0
    
    for track in score.tracks:
        new_track = Track(name=track.name, program=track.program, is_drum=track.is_drum)
        
        for note in track.notes:
            # Convert note start from ticks to seconds (original timing)
            note_start_secs = note.time * original_secs_per_tick
            note_end_secs = (note.time + note.duration) * original_secs_per_tick
            
            # Find beat position for start
            start_beat_idx, start_rel_pos = find_beat_and_position(note_start_secs)
            end_beat_idx, end_rel_pos = find_beat_and_position(note_end_secs)
            
            # Convert to new tick positions
            # beat_idx * ticks_per_beat + rel_pos * ticks_per_beat
            new_start_tick = (start_beat_idx + start_rel_pos) * ticks_per_beat
            new_end_tick = (end_beat_idx + end_rel_pos) * ticks_per_beat
            
            # Handle negative beat indices (notes before first beat)
            # We'll shift everything so first beat is at time 0
            # This will be handled by note_shift in chorderator
            
            # Quantize if requested
            if quantize_to_grid:
                new_start_tick = quantize_to_nearest_grid(new_start_tick, ticks_per_grid)
                new_end_tick = quantize_to_nearest_grid(new_end_tick, ticks_per_grid)
                if new_end_tick <= new_start_tick:
                    new_end_tick = new_start_tick + ticks_per_grid
                notes_quantized += 1
            
            new_duration = int(new_end_tick - new_start_tick)
            if new_duration < 1:
                new_duration = 1
            
            new_note = Note(
                time=int(new_start_tick),
                duration=new_duration,
                pitch=note.pitch,
                velocity=note.velocity
            )
            new_track.notes.append(new_note)
            notes_processed += 1
        
        new_track.notes.sort(key=lambda n: n.time)
        new_score.tracks.append(new_track)
    
    # Shift all notes so the minimum start time is >= 0
    all_notes = [n for t in new_score.tracks for n in t.notes]
    if all_notes:
        min_time = min(n.time for n in all_notes)
        if min_time < 0:
            shift_amount = -min_time
            # Round up to nearest bar
            ticks_per_bar = ticks_per_beat * beats_per_bar
            shift_bars = int(np.ceil(shift_amount / ticks_per_bar))
            actual_shift = shift_bars * ticks_per_bar
            
            print(f"Shifting notes by {shift_bars} bars ({actual_shift} ticks) to ensure positive times")
            
            for track in new_score.tracks:
                for note in track.notes:
                    note.time += actual_shift
    
    # Save the warped MIDI
    new_score.dump_midi(output_midi_path)
    
    print(f"Processed {notes_processed} notes")
    if quantize_to_grid:
        print(f"Quantized to 1/{grid_resolution} beat grid")
    print(f"Warped MIDI saved to: {output_midi_path}")
    
    return {
        'original_tempo': original_tempo,
        'new_tempo': new_tempo,
        'mean_ibi': mean_ibi,
        'bar_duration': tempo_info['bar_duration'],
        'notes_processed': notes_processed,
        'quantize_to_grid': quantize_to_grid,
        'grid_resolution': grid_resolution
    }


def find_first_downbeat_for_melody(beat_times, first_note_time, beats_per_bar=4):
    """
    Find which beat is the first downbeat that the melody starts in or after.
    
    Args:
        beat_times: Array of beat times in seconds
        first_note_time: Time of first note in seconds
        beats_per_bar: Number of beats per bar
    
    Returns:
        tuple: (downbeat_idx, downbeat_time, bar_number)
    """
    # Find the beat just before or at the first note
    beat_idx = 0
    for i, bt in enumerate(beat_times):
        if bt > first_note_time:
            beat_idx = max(0, i - 1)
            break
        beat_idx = i
    
    # Find the downbeat (bar start) for this beat
    # Assuming beat 0 is a downbeat
    bar_number = beat_idx // beats_per_bar
    downbeat_idx = bar_number * beats_per_bar
    
    # If first note is before the first beat, use bar 0
    if first_note_time < beat_times[0]:
        return 0, beat_times[0], 0
    
    return downbeat_idx, beat_times[downbeat_idx], bar_number


def detect_downbeat_phase(beat_times, midi_path, beats_per_bar=4, beat_subdivision=1):
    """
    Detect which index in beat_times corresponds to the first downbeat (bar 1 beat 1).

    Strategy: vocal note onsets tend to cluster on beat 1 of the bar more than other
    positions. We map every note onset to its nearest beat_times entry, tally which
    position-in-bar (beat_idx % entries_per_bar) accumulates the most onsets, and call
    that the downbeat. Instrumental intros are harmless because they contain no notes.

    Args:
        beat_times:      Array of beat onset times in seconds (from SingNet etc.)
        midi_path:       Path to the (single-track) vocal MIDI file.
        beats_per_bar:   Number of beats per bar (default 4 for 4/4).
        beat_subdivision: 1 = beat_times are quarter notes, 2 = eighth notes, etc.

    Returns:
        int: phase offset p such that beat_times[p] is the first downbeat.
             Slicing beat_times[p:] gives a beat sequence that starts on bar 1 beat 1.
    """
    import pretty_midi

    entries_per_bar = beats_per_bar * beat_subdivision

    if len(beat_times) < entries_per_bar:
        print("  [phase] Too few beat times — defaulting to phase=0")
        return 0

    pm = pretty_midi.PrettyMIDI(midi_path)
    note_onsets = [note.start for inst in pm.instruments for note in inst.notes]

    if not note_onsets:
        print("  [phase] No notes found — defaulting to phase=0")
        return 0

    # Count note onsets per bar-position across all positions 0..entries_per_bar-1
    phase_scores = np.zeros(entries_per_bar)
    for t in note_onsets:
        k = int(np.argmin(np.abs(beat_times - t)))
        phase_scores[k % entries_per_bar] += 1

    best_phase = int(np.argmax(phase_scores))

    print(f"  [phase] scores per bar position: {phase_scores.astype(int).tolist()}")
    print(f"  [phase] detected downbeat phase={best_phase} "
          f"→ beat_times[{best_phase}] = {beat_times[best_phase]:.3f}s")

    return best_phase


# pitch to key
pitch_to_key = {
    0: 'C',
    1: 'C#',
    2: 'D',
    3: 'D#',
    4: 'E',
    5: 'F',
    6: 'F#',
    7: 'G',
    8: 'G#',
    9: 'A',
    10: 'A#',
    11: 'B',
}
def get_segmentation(midi_obj):
    pass

def get_key(tokens):
    """
    Enhanced key detection using multiple methods:
    1. Most frequent pitch class
    2. Circle of fifths analysis
    3. Scale degree analysis
    
    Returns a key string that can be used with cdt.Key class
    """
    # Extract all pitch classes
    pitch_classes = []
    for token in tokens:
        if token.startswith('Pitch'):
            pitch = int(token.split('_')[1])
            pitch_classes.append(pitch % 12)
    
    if not pitch_classes:
        return 'C'
    
    # Method 1: Most frequent pitch class
    pitch_count = {}
    for pc in pitch_classes:
        note_name = pitch_to_key[pc]
        pitch_count[note_name] = pitch_count.get(note_name, 0) + 1
    
    most_common_pitch = max(pitch_count, key=pitch_count.get)
    
    # Method 2: Circle of fifths analysis for major/minor determination
    major_keys = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']
    minor_keys = ['A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F', 'C', 'G', 'D']
    
    # Count how many notes fit each key
    key_scores = {}
    
    for key in major_keys:
        key_pc = list(pitch_to_key.keys())[list(pitch_to_key.values()).index(key)]
        # Major scale pattern: 0, 2, 4, 5, 7, 9, 11
        major_scale = [(key_pc + interval) % 12 for interval in [0, 2, 4, 5, 7, 9, 11]]
        score = sum(1 for pc in pitch_classes if pc in major_scale)
        key_scores[f"{key}_major"] = score
    
    for key in minor_keys:
        key_pc = list(pitch_to_key.keys())[list(pitch_to_key.values()).index(key)]
        # Natural minor scale pattern: 0, 2, 3, 5, 7, 8, 10
        minor_scale = [(key_pc + interval) % 12 for interval in [0, 2, 3, 5, 7, 8, 10]]
        score = sum(1 for pc in pitch_classes if pc in minor_scale)
        key_scores[f"{key}_minor"] = score
    
    # Find the best key
    best_key = max(key_scores, key=key_scores.get)
    best_score = key_scores[best_key]
    
    # If the best key has significantly higher score, use it
    if best_score > len(pitch_classes) * 0.6:  # At least 60% of notes fit the key
        return best_key.split('_')[0]  # Return just the note name
    
    # Otherwise, fall back to most common pitch
    return most_common_pitch


def get_key_for_cdt(tokens, key_analysis=None):
    """
    Get key that can be directly used with cdt.Key class.
    Returns the appropriate cdt.Key attribute.
    
    Args:
        tokens: MIDI tokens
        key_analysis: Optional detailed key analysis result to use instead of basic detection
    """
    if key_analysis:
        key_name = key_analysis['key']
    else:
        key_name = get_key(tokens)
    
    # Map key names to cdt.Key attributes
    key_mapping = {
        'C': 'C',
        'C#': 'CSharp',
        'Db': 'DFlat', 
        'D': 'D',
        'D#': 'DSharp',
        'Eb': 'EFlat',
        'E': 'E',
        'F': 'F',
        'F#': 'FSharp',
        'Gb': 'GFlat',
        'G': 'G',
        'G#': 'GSharp',
        'Ab': 'AFlat',
        'A': 'A',
        'A#': 'ASharp',
        'Bb': 'BFlat',
        'B': 'B'
    }
    
    return key_mapping.get(key_name, 'C')


def get_mode_for_cdt(tokens, key_analysis=None):
    """
    Get mode that can be directly used with cdt.Mode class.
    Returns the appropriate cdt.Mode attribute.
    
    Args:
        tokens: MIDI tokens
        key_analysis: Optional detailed key analysis result to use instead of basic detection
    """
    if key_analysis:
        mode_name = key_analysis['mode']
    else:
        # Fallback to basic detection
        key_analysis = get_detailed_key_analysis(tokens)
        mode_name = key_analysis['mode']
    
    # Map mode names to cdt.Mode attributes
    mode_mapping = {
        'major': 'MAJOR',
        'minor': 'MINOR',
        'maj': 'MAJOR',
        'min': 'MINOR'
    }
    
    return mode_mapping.get(mode_name, 'MAJOR')


def get_detailed_key_analysis(tokens):
    """
    Detailed key analysis with confidence scores and mode detection.
    """
    # Extract all pitch classes
    pitch_classes = []
    for token in tokens:
        if token.startswith('Pitch'):
            pitch = int(token.split('_')[1])
            pitch_classes.append(pitch % 12)
    
    if not pitch_classes:
        return {'key': 'C', 'mode': 'major', 'confidence': 0.0, 'analysis': 'No notes found'}
    
    # Count pitch class occurrences
    pc_count = {}
    for pc in pitch_classes:
        pc_count[pc] = pc_count.get(pc, 0) + 1
    
    # Major and minor key analysis
    major_keys = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']
    minor_keys = ['A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F', 'C', 'G', 'D']
    
    key_analysis = {}
    
    # Analyze major keys
    for key in major_keys:
        key_pc = list(pitch_to_key.keys())[list(pitch_to_key.values()).index(key)]
        major_scale = [(key_pc + interval) % 12 for interval in [0, 2, 4, 5, 7, 9, 11]]
        
        # Count notes that fit the scale
        scale_notes = sum(1 for pc in pitch_classes if pc in major_scale)
        total_notes = len(pitch_classes)
        confidence = scale_notes / total_notes if total_notes > 0 else 0
        
        # Bonus for having the tonic
        tonic_bonus = 0.1 if key_pc in pitch_classes else 0
        
        key_analysis[f"{key}_major"] = {
            'key': key,
            'mode': 'major',
            'confidence': min(confidence + tonic_bonus, 1.0),
            'scale_notes': scale_notes,
            'total_notes': total_notes
        }
    
    # Analyze minor keys
    for key in minor_keys:
        key_pc = list(pitch_to_key.keys())[list(pitch_to_key.values()).index(key)]
        # Natural minor scale pattern: 0, 2, 3, 5, 7, 8, 10
        minor_scale = [(key_pc + interval) % 12 for interval in [0, 2, 3, 5, 7, 8, 10]]
        
        scale_notes = sum(1 for pc in pitch_classes if pc in minor_scale)
        total_notes = len(pitch_classes)
        confidence = scale_notes / total_notes if total_notes > 0 else 0
        
        # Bonus for having the tonic
        tonic_bonus = 0.1 if key_pc in pitch_classes else 0
        
        # Additional bonus for minor third (characteristic of minor keys)
        minor_third = (key_pc + 3) % 12
        minor_third_bonus = 0.05 if minor_third in pitch_classes else 0
        
        # Additional bonus for minor sixth (another characteristic of minor keys)
        minor_sixth = (key_pc + 8) % 12
        minor_sixth_bonus = 0.05 if minor_sixth in pitch_classes else 0
        
        key_analysis[f"{key}_minor"] = {
            'key': key,
            'mode': 'minor',
            'confidence': min(confidence + tonic_bonus + minor_third_bonus + minor_sixth_bonus, 1.0),
            'scale_notes': scale_notes,
            'total_notes': total_notes
        }
    
    # Find the best key
    best_key_name = max(key_analysis, key=lambda k: key_analysis[k]['confidence'])
    best_analysis = key_analysis[best_key_name]
    
    # Handle ties in confidence scores - prefer C major when tied
    best_confidence = best_analysis['confidence']
    tied_keys = [k for k, v in key_analysis.items() if abs(v['confidence'] - best_confidence) < 0.001]
    
    if len(tied_keys) > 1:
        # Priority order: C major > other major keys > minor keys
        if 'C_major' in tied_keys:
            best_key_name = 'C_major'
            best_analysis = key_analysis[best_key_name]
        else:
            # Check if there are both major and minor versions of the same key
            major_keys_in_tie = [k for k in tied_keys if k.endswith('_major')]
            minor_keys_in_tie = [k for k in tied_keys if k.endswith('_minor')]
            
            # If we have both major and minor of the same key, prefer major
            for minor_key in minor_keys_in_tie:
                key_name = minor_key.replace('_minor', '')
                corresponding_major = f"{key_name}_major"
                if corresponding_major in major_keys_in_tie:
                    best_key_name = corresponding_major
                    best_analysis = key_analysis[best_key_name]
                    break
            
            # If no direct major/minor pair, prefer any major key over minor
            if best_key_name.endswith('_minor') and major_keys_in_tie:
                # Find the major key with highest confidence
                best_major = max(major_keys_in_tie, key=lambda k: key_analysis[k]['confidence'])
                best_key_name = best_major
                best_analysis = key_analysis[best_key_name]
    
    # Get top 3 candidates
    sorted_keys = sorted(key_analysis.items(), key=lambda x: x[1]['confidence'], reverse=True)
    top_candidates = sorted_keys[:3]
    
    return {
        'key': best_analysis['key'],
        'mode': best_analysis['mode'],
        'confidence': best_analysis['confidence'],
        'analysis': f"Best fit: {best_analysis['key']} {best_analysis['mode']} (confidence: {best_analysis['confidence']:.2f})",
        'top_candidates': [(name, data['confidence']) for name, data in top_candidates],
        'all_analysis': key_analysis
    }


def analyze_chord_notes(notes, key=None):
    """
    Analyze a list of notes and return possible chord types with their root notes.
    Now considers key context for better analysis.
    
    Args:
        notes: List of MIDI note numbers or note names
        key: Optional key context for better analysis
    
    Returns:
        List of tuples (root_note, chord_type, confidence_score)
    """
    # Convert notes to pitch classes (0-11)
    pitch_classes = set()
    
    for note in notes:
        if isinstance(note, int):
            # MIDI note number
            pitch_classes.add(note % 12)
        elif isinstance(note, str):
            # Note name like 'C', 'C#', 'Db', etc.
            if note in pitch_to_key.values():
                # Find the MIDI number for this note
                for midi_num, note_name in pitch_to_key.items():
                    if note_name == note:
                        pitch_classes.add(midi_num)
                        break
    
    if len(pitch_classes) < 2:
        return []
    
    # Define chord patterns
    chord_patterns = {
        # Major chords
        'major': [0, 4, 7],
        'major7': [0, 4, 7, 11],
        'major9': [0, 4, 7, 11, 2],
        'major6': [0, 4, 7, 9],
        'major6/9': [0, 4, 7, 9, 2],
        
        # Minor chords
        'minor': [0, 3, 7],
        'minor7': [0, 3, 7, 10],
        'minor9': [0, 3, 7, 10, 2],
        'minor6': [0, 3, 7, 9],
        'minor6/9': [0, 3, 7, 9, 2],
        'minor_major7': [0, 3, 7, 11],
        
        # Dominant chords
        'dominant7': [0, 4, 7, 10],
        'dominant9': [0, 4, 7, 10, 2],
        'dominant11': [0, 4, 7, 10, 2, 5],
        'dominant13': [0, 4, 7, 10, 2, 5, 9],
        
        # Diminished chords
        'diminished': [0, 3, 6],
        'diminished7': [0, 3, 6, 9],
        'half_diminished7': [0, 3, 6, 10],
        
        # Augmented chords
        'augmented': [0, 4, 8],
        'augmented7': [0, 4, 8, 10],
        
        # Suspended chords
        'sus2': [0, 2, 7],
        'sus4': [0, 5, 7],
        'sus2/7': [0, 2, 7, 10],
        'sus4/7': [0, 5, 7, 10],
        
        # Power chords
        'power_chord': [0, 7],
        'power_octave': [0, 12],  # Same note, different octave
        'full_octave': [0, 7, 12],  # Root, fifth, octave
        
        # Cluster chords
        'cluster': [0, 1, 2],  # Adjacent notes
        'cluster_wide': [0, 1, 2, 3],  # Wider cluster
        
        # Inversions
        'first_inversion': [0, 3, 8],  # Major first inversion
        'second_inversion': [0, 5, 9],  # Major second inversion
        
        # Root note only
        'root_note': [0],
    }
    
    # Convert pitch classes to list for easier manipulation
    pitch_list = sorted(list(pitch_classes))
    
    # Get key context for better analysis
    key_pc = None
    key_scale = None
    if key and key in pitch_to_key.values():
        key_pc = list(pitch_to_key.keys())[list(pitch_to_key.values()).index(key)]
        # Assume major scale for now (could be enhanced to detect mode)
        key_scale = [(key_pc + interval) % 12 for interval in [0, 2, 4, 5, 7, 9, 11]]
    
    # Find possible chords
    possible_chords = []
    
    for root in range(12):
        # Try each root note
        for chord_name, pattern in chord_patterns.items():
            # Transpose the pattern to the root
            transposed_pattern = [(p + root) % 12 for p in pattern]
            
            # Calculate how many notes match
            matches = len(set(transposed_pattern) & set(pitch_list))
            total_notes = len(pattern)
            
            # Calculate confidence score
            if matches >= 2:  # At least 2 notes must match
                confidence = matches / total_notes
                
                # Bonus for having the root note
                if root in pitch_list:
                    confidence += 0.2
                
                # Bonus for having the third (major/minor indicator)
                if (root + 4) % 12 in pitch_list or (root + 3) % 12 in pitch_list:
                    confidence += 0.1
                
                # Bonus for having the fifth
                if (root + 7) % 12 in pitch_list:
                    confidence += 0.1
                
                # KEY CONTEXT BONUS: Prefer chords that fit the key
                if key_scale and root in key_scale:
                    confidence += 0.15  # Significant bonus for chords in key
                
                # Additional bonus for chord tones that are in the key scale
                if key_scale:
                    chord_tones_in_key = sum(1 for pc in transposed_pattern if pc in key_scale)
                    key_bonus = (chord_tones_in_key / len(transposed_pattern)) * 0.1
                    confidence += key_bonus
                
                # Cap confidence at 1.0
                confidence = min(confidence, 1.0)
                
                root_note = pitch_to_key[root]
                possible_chords.append((root_note, chord_name, confidence))
    
    # Sort by confidence score (highest first)
    possible_chords.sort(key=lambda x: x[2], reverse=True)
    
    # Filter out very low confidence matches
    possible_chords = [chord for chord in possible_chords if chord[2] >= 0.3]
    
    return possible_chords


def get_chord_analysis(tokens, key=None):
    """
    Analyze chords from tokenized MIDI data and return possible chord types.
    
    Args:
        tokens: List of tokens from MIDI tokenizer
        key: Optional key context
    
    Returns:
        Dictionary with chord analysis results
    """
    # Extract notes from tokens
    notes = []
    for token in tokens:
        if token.startswith('Pitch'):
            pitch = int(token.split('_')[1])
            notes.append(pitch)
    
    if not notes:
        return {'error': 'No notes found in tokens'}
    
    # Analyze chords
    chord_analysis = analyze_chord_notes(notes, key)
    
    # Get the most common pitch as key if not provided
    if key is None:
        key = get_key(tokens)
    
    # Group by root note
    chords_by_root = {}
    for root, chord_type, confidence in chord_analysis:
        if root not in chords_by_root:
            chords_by_root[root] = []
        chords_by_root[root].append((chord_type, confidence))
    
    # Sort chords within each root by confidence
    for root in chords_by_root:
        chords_by_root[root].sort(key=lambda x: x[1], reverse=True)
    
    return {
        'key': key,
        'notes': sorted(list(set([note % 12 for note in notes]))),
        'note_names': [pitch_to_key[note % 12] for note in set(notes)],
        'possible_chords': chord_analysis,
        'chords_by_root': chords_by_root,
        'top_chords': chord_analysis[:5]  # Top 5 most likely chords
    }


def get_advanced_chord_analysis(tokens, key=None, style_context=None):
    """
    Advanced chord analysis with style classification and AccoMontage integration.
    
    Args:
        tokens: List of tokens from MIDI tokenizer
        key: Optional key context
        style_context: Optional style context for better analysis
    
    Returns:
        Dictionary with comprehensive chord analysis including style suggestions
    """
    # Get basic chord analysis
    basic_analysis = get_chord_analysis(tokens, key)
    
    if 'error' in basic_analysis:
        return basic_analysis
    
    # Map chord types to AccoMontage chord styles
    chord_style_mapping = {
        'major': 'standard',
        'minor': 'standard',
        'major7': 'seventh',
        'minor7': 'seventh',
        'dominant7': 'seventh',
        'sus2': 'sus2',
        'sus4': 'sus4',
        'power_chord': 'power-chord',
        'power_octave': 'power-octave',
        'full_octave': 'full-octave',
        'cluster': 'cluster',
        'cluster_wide': 'cluster',
        'first_inversion': 'first-inversion',
        'second_inversion': 'second-inversion',
        'root_note': 'root-note',
        'augmented': 'classy',
        'diminished': 'classy',
        'diminished7': 'classy',
        'half_diminished7': 'classy',
        'minor_major7': 'classy',
        'major9': 'emotional',
        'minor9': 'emotional',
        'dominant9': 'emotional',
        'dominant11': 'emotional',
        'dominant13': 'emotional',
    }
    
    # Analyze chord styles
    chord_styles = {}
    for root, chord_type, confidence in basic_analysis['possible_chords']:
        if chord_type in chord_style_mapping:
            style = chord_style_mapping[chord_type]
            if style not in chord_styles:
                chord_styles[style] = []
            chord_styles[style].append((root, chord_type, confidence))
    
    # Sort styles by average confidence
    style_rankings = []
    for style, chords in chord_styles.items():
        avg_confidence = sum(conf for _, _, conf in chords) / len(chords)
        max_confidence = max(conf for _, _, conf in chords)
        style_rankings.append((style, avg_confidence, max_confidence, len(chords)))
    
    style_rankings.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    
    # Suggest progression styles based on chord analysis
    progression_style_suggestions = []
    
    # Count different chord types
    major_count = sum(1 for _, chord_type, _ in basic_analysis['possible_chords'] 
                     if 'major' in chord_type and 'minor' not in chord_type)
    minor_count = sum(1 for _, chord_type, _ in basic_analysis['possible_chords'] 
                     if 'minor' in chord_type)
    seventh_count = sum(1 for _, chord_type, _ in basic_analysis['possible_chords'] 
                       if '7' in chord_type)
    sus_count = sum(1 for _, chord_type, _ in basic_analysis['possible_chords'] 
                   if 'sus' in chord_type)
    
    # Suggest progression styles
    if seventh_count > 2:
        progression_style_suggestions.append(('r&b', 0.8))
    if sus_count > 1:
        progression_style_suggestions.append(('pop', 0.7))
    if minor_count > major_count:
        progression_style_suggestions.append(('dark', 0.6))
    if major_count > minor_count and seventh_count < 2:
        progression_style_suggestions.append(('pop', 0.7))
    
    # Default to pop if no clear pattern
    if not progression_style_suggestions:
        progression_style_suggestions.append(('pop', 0.5))
    
    progression_style_suggestions.sort(key=lambda x: x[1], reverse=True)
    
    # Add AccoMontage integration suggestions
    accomontage_suggestions = {
        'chord_styles': [style for style, _, _, _ in style_rankings[:3]],
        'progression_styles': [style for style, _ in progression_style_suggestions[:3]],
        'recommended_chord_style': style_rankings[0][0] if style_rankings else 'standard',
        'recommended_progression_style': progression_style_suggestions[0][0] if progression_style_suggestions else 'pop'
    }
    
    return {
        **basic_analysis,
        'chord_styles': chord_styles,
        'style_rankings': style_rankings,
        'progression_style_suggestions': progression_style_suggestions,
        'accomontage_suggestions': accomontage_suggestions
    }


def calculate_chorderator_bars(midi_path, tempo=None, note_shift=0):
    """
    Calculate the total bars exactly as chorderator will see them.
    
    This mimics chorderator's __construct_melo_sequence logic:
    1. Quantize note positions to 16th notes
    2. Find max_end position
    3. Apply fix_end to round up to multiple of 4 bars
    
    Args:
        midi_path: Path to MIDI file
        tempo: Tempo to use (if None, reads from MIDI)
        note_shift: Note shift in 16th note positions (same as chorderator)
    
    Returns:
        dict: {
            'max_end_positions': int (max note end in 16th note positions),
            'fixed_end_positions': int (rounded up to 4-bar boundary),
            'fixed_end_bars': int (total bars chorderator will see),
            'tempo': float,
            'unit': float (seconds per 16th note)
        }
    """
    from pretty_midi import PrettyMIDI
    
    midi = PrettyMIDI(midi_path)
    
    # Get tempo (same logic as chorderator)
    if tempo is None:
        tempo = midi.get_tempo_changes()[1][0]
    
    # Unit = seconds per 16th note (same as chorderator)
    unit = 60 / tempo / 4
    
    def quantize_note(time, unit):
        """Quantize time to nearest 16th note position (same as chorderator)."""
        base = time // unit
        return int(base if time % unit < unit / 2 else base + 1)
    
    def fix_end(max_end_bars):
        """Round up to multiple of 4 bars (same as chorderator)."""
        max_end_bars = int(max_end_bars)
        return max_end_bars if max_end_bars % 4 == 0 else int(((max_end_bars // 4) + 1) * 4)
    
    # Find max note end position
    all_end_positions = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            end_pos = quantize_note(note.end, unit) - note_shift
            if end_pos > 0:
                all_end_positions.append(end_pos)
    
    if not all_end_positions:
        return {
            'max_end_positions': 0,
            'fixed_end_positions': 64,  # minimum 4 bars
            'fixed_end_bars': 4,
            'tempo': tempo,
            'unit': unit
        }
    
    max_end = max(all_end_positions)
    max_end_bars = max_end / 16
    fixed_end_bars = fix_end(max_end_bars)
    fixed_end_positions = fixed_end_bars * 16
    
    return {
        'max_end_positions': max_end,
        'fixed_end_positions': fixed_end_positions,
        'fixed_end_bars': fixed_end_bars,
        'tempo': tempo,
        'unit': unit
    }


def analyze_midi_structure_from_file(midi_path):
    """
    Analyze MIDI structure directly from file to get accurate bar counts.
    
    Args:
        midi_path: Path to the MIDI file
    
    Returns:
        dict with empty_bars, content_bars, note_shift, etc.
    """
    score = Score(midi_path, ttype="tick")
    tpq = score.ticks_per_quarter
    ticks_per_bar = tpq * 4  # 4/4 time
    
    # Get all notes
    all_notes = []
    for track in score.tracks:
        all_notes.extend(track.notes)
    
    if not all_notes:
        return {
            'total_bars': 0,
            'empty_bars': 0,
            'content_bars': 4,
            'note_shift': 0,
            'segmentation': 'A4',
        }
    
    # Find first and last note positions
    first_note_tick = min(n.start for n in all_notes)
    last_note_tick = max(n.end for n in all_notes)

    # Calculate bars
    first_note_bar = first_note_tick / ticks_per_bar
    last_note_bar = last_note_tick / ticks_per_bar

    # Empty bars = floor of first note bar (complete empty bars before first note)
    empty_bars = int(first_note_bar)

    # Content bars = from first note bar to last note bar
    # Use floor to ensure we don't claim more bars than we have content
    content_bars = int(last_note_bar) - empty_bars

    # Ensure content_bars is at least 4
    if content_bars < 4:
        content_bars = 4

    # Note shift: number of leading 16th-note positions chorderator should skip.
    # Must align to the actual first-note position (in 16th notes), NOT to whole
    # bars. If the melody has an anacrusis pickup (e.g., first_note_bar=6.75),
    # `empty_bars * 16` would leave a fractional bar of pickup attached to the
    # piano roll AccoMontage feeds into dp_search, causing it to match templates
    # that don't fit the phrase structure and produce gappy accompaniments.
    ticks_per_16th = ticks_per_bar / 16
    note_shift = int(round(first_note_tick / ticks_per_16th))
    
    # Calculate segmentation (round DOWN to avoid empty phrases)
    segmentation = calculate_segmentation(content_bars)
    
    return {
        'total_bars': int(last_note_bar) + 1,
        'empty_bars': empty_bars,
        'content_bars': content_bars,
        'note_shift': note_shift,
        'segmentation': segmentation,
        'first_note_bar': first_note_bar,
        'last_note_bar': last_note_bar,
    }


def analyze_midi_structure(tokens):
    """
    Analyze MIDI structure to determine note_shift and segmentation
    
    NOTE: This token-based analysis may not be accurate. 
    Consider using analyze_midi_structure_from_file() for better accuracy.
    
    Args:
        tokens: List of tokens from tokenizer
    
    Returns:
        dict: {
            'total_bars': int,
            'empty_bars': int,
            'content_bars': int,
            'note_shift': int,
            'segmentation': str,
            'first_note_position': int,
            'bar_positions': list
        }
    """
    # Find all bar positions
    bar_positions = [i for i, token in enumerate(tokens) if token.startswith('Bar_None')]
    total_bars = len(bar_positions)
    
    # Find first note position
    first_note_position = None
    for i, token in enumerate(tokens):
        if token.startswith('Pitch_'):
            first_note_position = i
            break


    empty_bars = 0
    # Calculate empty bars before first note
    # A bar is truly empty if the NEXT bar also starts before the first note
    # (meaning no Pitch tokens exist in that bar)
    if first_note_position is not None and len(bar_positions) >= 2:
        for i in range(len(bar_positions) - 1):
            # If the next bar also starts before the first note, this bar is empty
            if bar_positions[i + 1] <= first_note_position:
                empty_bars += 1
            else:
                # This bar contains the first note
                break
    
    # Calculate content bars (bars with actual content)
    content_bars = total_bars - empty_bars
    
    # Calculate note_shift (empty_bars * 16 positions per bar)
    note_shift = empty_bars * 16
    
    # Calculate segmentation based on content_bars
    segmentation = calculate_segmentation(content_bars)
    
    return {
        'total_bars': total_bars,
        'empty_bars': empty_bars,
        'content_bars': content_bars,
        'note_shift': note_shift,
        'segmentation': segmentation,
        'first_note_position': first_note_position,
        'bar_positions': bar_positions
    }


def get_valid_bar_count(content_bars, round_up=False):
    """
    Get the nearest valid bar count that chorderator supports.
    
    Chorderator supports any phrase length that is a multiple of 4 bars (minimum 4 bars).
    
    Args:
        content_bars: Actual number of bars with content
        round_up: If True, round UP to nearest multiple of 4.
                  If False (default), round DOWN to avoid empty phrases.
    
    Returns:
        int: Nearest valid bar count (multiple of 4)
    """
    if content_bars <= 0:
        return 4
    
    if content_bars <= 4:
        return 4
    
    if round_up:
        # Round UP to nearest multiple of 4
        return ((content_bars + 3) // 4) * 4
    else:
        # Round DOWN to nearest multiple of 4 (safer - avoids empty phrases)
        return (content_bars // 4) * 4


def get_segmentation_splits(content_bars):
    """
    Get the segmentation splits for content_bars.
    
    Args:
        content_bars: Number of bars with actual content
    
    Returns:
        list: List of (start_bar, end_bar, segment_length) tuples
              segment_length is the padded length for that segment
    """
    if content_bars <= 0:
        return [(0, 4, 4)]
    
    # Simply pad to the valid bar count
    padded_length = get_valid_bar_count(content_bars)
    return [(0, content_bars, padded_length)]


def calculate_segmentation(content_bars):
    """
    Calculate segmentation pattern based on content bars.
    
    Chorderator supports any phrase length that is a multiple of 4 bars (minimum 4 bars).
    This function pads content_bars UP to the nearest multiple of 4.
    
    Args:
        content_bars: Number of bars with actual content
    
    Returns:
        str: Segmentation pattern (e.g., 'A8B8A8B8', 'A4', 'A8', 'A8B4')
    """
    if content_bars <= 0:
        return 'A4'  # Default fallback
    
    # Get the valid (padded) total — round UP to match chorderator's fix_end behaviour
    valid_total = get_valid_bar_count(content_bars, round_up=True)
    
    if valid_total <= 4:
        print(f"Content bars: {content_bars} -> Segmentation: A4 (total: 4)")
        return 'A4'
    
    if valid_total <= 8:
        print(f"Content bars: {content_bars} -> Segmentation: A8 (total: 8)")
        return 'A8'
    
    # Build segments using 8s and 4s
    segments = []
    remaining = valid_total
    
    while remaining >= 8:
        segments.append(8)
        remaining -= 8
    
    if remaining == 4:
        segments.append(4)
    
    # Build segmentation string with alternating A/B labels
    seg_parts = []
    for i, length in enumerate(segments):
        label = chr(ord('A') + (i % 2))  # Alternate A, B, A, B, ...
        seg_parts.append(f"{label}{length}")
    
    result = ''.join(seg_parts)
    total = sum(segments)
    
    print(f"Content bars: {content_bars} -> Segmentation: {result} (total: {total})")
    return result


def get_auto_config(tokens, midi_path=None, tempo=None):
    """
    Get automatic configuration for AccoMontage based on MIDI analysis

    Args:
        tokens: List of tokens from tokenizer
        midi_path: Optional path to MIDI file for more accurate analysis
        tempo: Optional tempo (BPM). If provided, used for quantization instead of reading from MIDI.

    Returns:
        dict: {
            'note_shift': int,
            'segmentation': str,
            'analysis': dict (from analyze_midi_structure)
        }
    """
    # Use file-based analysis if midi_path is provided (more accurate)
    if midi_path is not None:
        analysis = analyze_midi_structure_from_file(midi_path)

        # Calculate what chorderator will actually see
        # This is crucial: chorderator rounds UP to 4-bar boundaries
        chorderator_bars = calculate_chorderator_bars(midi_path, tempo=tempo, note_shift=analysis['note_shift'])
        
        # Use chorderator's fixed_end_bars for segmentation
        # This ensures our segmentation matches what chorderator will actually process
        actual_bars_for_segmentation = chorderator_bars['fixed_end_bars']
        
        # Recalculate segmentation based on what chorderator sees
        segmentation = calculate_segmentation_for_chorderator(actual_bars_for_segmentation)
        
        analysis['chorderator_bars'] = chorderator_bars
        analysis['segmentation'] = segmentation
    else:
        analysis = analyze_midi_structure(tokens)
        segmentation = analysis['segmentation']
    
    return {
        'note_shift': analysis['note_shift'],
        'segmentation': segmentation,
        'analysis': analysis
    }


def calculate_segmentation_for_chorderator(total_bars):
    """
    Calculate segmentation pattern that exactly matches the given total bars.
    
    This is used when we know exactly how many bars chorderator will process
    (from calculate_chorderator_bars).
    
    Args:
        total_bars: Total bars that chorderator will see (already rounded to 4)
    
    Returns:
        str: Segmentation pattern that covers exactly total_bars
    """
    if total_bars <= 0:
        return 'A4'
    
    # Ensure it's a multiple of 4
    if total_bars % 4 != 0:
        total_bars = ((total_bars // 4) + 1) * 4
    
    if total_bars <= 4:
        print(f"Chorderator bars: {total_bars} -> Segmentation: A4")
        return 'A4'
    
    if total_bars <= 8:
        print(f"Chorderator bars: {total_bars} -> Segmentation: A8")
        return 'A8'
    
    # Build segments using 8s and 4s
    segments = []
    remaining = total_bars
    
    while remaining >= 8:
        segments.append(8)
        remaining -= 8
    
    if remaining == 4:
        segments.append(4)
    
    # Build segmentation string with alternating A/B labels
    seg_parts = []
    for i, length in enumerate(segments):
        label = chr(ord('A') + (i % 2))  # Alternate A, B, A, B, ...
        seg_parts.append(f"{label}{length}")
    
    result = ''.join(seg_parts)
    actual_total = sum(segments)
    
    print(f"Chorderator bars: {total_bars} -> Segmentation: {result} (total: {actual_total})")
    return result


def _build_tempo_map(midi):
    """Build tempo map: list of (abs_ticks, us_per_beat).
    Default tempo is 500000 us/beat until first set_tempo.
    """
    import mido
    ticks_per_beat = midi.ticks_per_beat
    merged = mido.merge_tracks(midi.tracks)
    tempos = []
    current_tick = 0
    current_tempo = 500000  # default 120bpm

    # Start with initial tempo at tick 0
    tempos.append((0, current_tempo))

    for msg in merged:
        current_tick += msg.time
        if msg.type == 'set_tempo':
            current_tempo = msg.tempo
            # Avoid duplicates at same tick
            if tempos and tempos[-1][0] == current_tick:
                tempos[-1] = (current_tick, current_tempo)
            else:
                tempos.append((current_tick, current_tempo))
    return tempos, ticks_per_beat


def _ticks_to_seconds(ticks, tempo_map, tpq):
    """Convert absolute ticks to seconds using tempo map (list of (tick, us_per_beat))."""
    seconds = 0.0
    last_tick = 0
    last_tempo = tempo_map[0][1] if tempo_map else 500000
    for i in range(1, len(tempo_map)):
        t_tick, t_tempo = tempo_map[i]
        if ticks <= t_tick:
            seg_ticks = ticks - last_tick
            seconds += (seg_ticks / tpq) * (last_tempo / 1_000_000.0)
            return seconds
        # full segment
        seg_ticks = t_tick - last_tick
        seconds += (seg_ticks / tpq) * (last_tempo / 1_000_000.0)
        last_tick = t_tick
        last_tempo = t_tempo
    # after last tempo change
    seg_ticks = ticks - last_tick
    seconds += (seg_ticks / tpq) * (last_tempo / 1_000_000.0)
    return seconds


def _quality_to_label(root: str, chord_type: str) -> str:
    ct = chord_type.lower()
    if ct == 'major':
        return root
    if ct == 'minor':
        return f"{root}:min"
    if ct == 'major7':
        return f"{root}:maj7"
    if ct == 'minor7':
        return f"{root}:min7"
    if ct == 'dominant7':
        return f"{root}:7"
    if ct == 'diminished':
        return f"{root}:dim"
    if ct == 'half_diminished7':
        return f"{root}:ø"
    if ct == 'diminished7':
        return f"{root}:dim7"
    if ct == 'augmented':
        return f"{root}:aug"
    if ct == 'sus2':
        return f"{root}:sus2"
    if ct == 'sus4':
        return f"{root}:sus4"
    if ct == 'major9':
        return f"{root}:maj9"
    if ct == 'minor9':
        return f"{root}:min9"
    if ct == 'dominant9':
        return f"{root}:9"
    if ct == 'dominant11':
        return f"{root}:11"
    if ct == 'dominant13':
        return f"{root}:13"
    if 'minor' in ct:
        return f"{root}:min"
    if 'sus' in ct:
        return f"{root}:sus"
    if 'aug' in ct:
        return f"{root}:aug"
    if 'dim' in ct:
        return f"{root}:dim"
    return root


def _is_diatonic(root_note: str, chord_type: str, key_name: str, mode: str) -> bool:
    try:
        root_pc = list(pitch_to_key.keys())[list(pitch_to_key.values()).index(root_note)]
    except ValueError:
        return False
    if key_name not in pitch_to_key.values():
        return True
    key_pc = list(pitch_to_key.keys())[list(pitch_to_key.values()).index(key_name)]
    if (mode or '').lower().startswith('maj'):
        degree_to_quality = {0: 'major', 2: 'minor', 4: 'minor', 5: 'major', 7: 'major', 9: 'minor', 11: 'diminished'}
    else:
        degree_to_quality = {0: 'minor', 2: 'diminished', 3: 'major', 5: 'minor', 7: 'minor', 8: 'major', 10: 'major'}
    semitone = (root_pc - key_pc) % 12
    expected_quality = degree_to_quality.get(semitone)
    if expected_quality is None:
        return False
    ct = chord_type.lower()
    # Treat root-only detection as acceptable (assume tonic chord tone context)
    if ct == 'root_note':
        return True
    if ct.startswith('sus'):
        return expected_quality in ('major', 'minor')
    if expected_quality == 'major':
        if ct in ('major', 'major7', 'major9'):
            return True
        if ct in ('dominant7', 'dominant9', 'dominant11', 'dominant13'):
            return semitone == 7
        return False
    if expected_quality == 'minor':
        return ct in ('minor', 'minor7', 'minor9')
    if expected_quality == 'diminished':
        return ct in ('diminished', 'half_diminished7', 'diminished7')
    return False


def _notes_to_chord_root_and_type(note_numbers, key_name=None, mode=None):
    if not note_numbers:
        return None
    # Use key to bias candidates if available
    possible = analyze_chord_notes(list(note_numbers), key=key_name) if key_name else analyze_chord_notes(list(note_numbers))
    if not possible:
        return None
    # If key provided, prefer diatonic candidates when close in score, and avoid always picking tonic
    if key_name is not None and mode is not None:
        # prefer diatonic candidates; first, if same-root diatonic exists, choose best of them
        # otherwise choose best diatonic within a reasonable margin; otherwise fallback to top
        # find top score
        top_score = possible[0][2]
        # best diatonic overall
        diatonic_all = [(r, ct, sc) for (r, ct, sc) in possible if _is_diatonic(r, ct, key_name, mode)]
        if diatonic_all:
            # prefer same-root diatonic when available
            # get root of top candidate for tie-breaking
            top_root = possible[0][0]
            same_root_diatonic = [(r, ct, sc) for (r, ct, sc) in diatonic_all if r == top_root]
            if same_root_diatonic:
                r, ct, _ = same_root_diatonic[0]
                return r, ct
            # otherwise take highest-score diatonic, allowing larger margin
            diatonic_all.sort(key=lambda x: x[2], reverse=True)
            # try prefer non-tonic root if close to top, to avoid always C
            non_tonic = [(r, ct, sc) for (r, ct, sc) in diatonic_all if r != key_name]
            if non_tonic:
                non_tonic.sort(key=lambda x: x[2], reverse=True)
                r2, ct2, sc2 = non_tonic[0]
                if (top_score - sc2) <= 0.18 or not _is_diatonic(possible[0][0], possible[0][1], key_name, mode):
                    return r2, ct2
            r, ct, sc = diatonic_all[0]
            if (top_score - sc) <= 0.25 or not _is_diatonic(possible[0][0], possible[0][1], key_name, mode):
                return r, ct
    r, ct, _ = possible[0]
    return r, ct


def export_chords_txt(midi_file_path, output_txt_path=None, key_name=None, mode=None):
    """
    Extract chord track and export as a text file with lines:
    start_sec end_sec chord_label

    - start/end in seconds with 3 decimals
    - chord_label like 'C', 'D:min', or 'N'
    - saved next to the MIDI file if output_txt_path not provided
    """
    # Load with symusic for reliable note parsing (ticks)
    score = Score(midi_file_path, ttype='tick')
    if len(score.tracks) < 2:
        # Still produce an empty file with N
        from pathlib import Path
        out = output_txt_path or str(Path(midi_file_path).with_suffix('')) + '_chords.txt'
        with open(out, 'w', encoding='utf-8') as f:
            f.write('0.000 0.000 N\n')
        return out

    # Identify chord track
    chord_idx = None
    for i, track in enumerate(score.tracks):
        name = (track.name or '').lower()
        if 'chord' in name or i == 1:
            chord_idx = i
            break
    if chord_idx is None:
        chord_idx = 1 if len(score.tracks) > 1 else 0

    chord_notes = score.tracks[chord_idx].notes

    # Detect key/mode if not provided
    if key_name is None or mode is None:
        pseudo_tokens = []
        for tr in score.tracks:
            for n in tr.notes:
                pseudo_tokens.append(f"Pitch_{n.pitch}")
        analysis = get_detailed_key_analysis(pseudo_tokens)
        key_name = key_name or analysis['key']
        mode = mode or analysis['mode']

    # Build event list (start/end) in ticks
    events = []  # (tick, type, note)
    for n in chord_notes:
        events.append((n.start, 1, n.pitch))  # 1 = note_on
        events.append((n.end, 0, n.pitch))    # 0 = note_off
    if not events:
        from pathlib import Path
        out = output_txt_path or str(Path(midi_file_path).with_suffix('')) + '_chords.txt'
        with open(out, 'w', encoding='utf-8') as f:
            f.write('0.000 0.000 N\n')
        return out

    # Sort by time, note_off before note_on at same tick to avoid zero-length segments
    events.sort(key=lambda x: (x[0], x[1]))

    # Prepare tempo map using mido
    import mido
    midi = mido.MidiFile(midi_file_path)
    tempo_map, tpq = _build_tempo_map(midi)

    # Sweep to build segments
    active = set()
    segments = []  # (start_tick, end_tick, label, out_of_key: bool)
    last_tick = events[0][0]
    for tick, typ, note in events:
        if tick > last_tick:
            rot = _notes_to_chord_root_and_type(active, key_name, mode)
            if rot is None:
                label = 'N'
                out_of_key = False
            else:
                root, chord_type = rot
                label = _quality_to_label(root, chord_type)
                out_of_key = not _is_diatonic(root, chord_type, key_name, mode)
            segments.append((last_tick, tick, label, out_of_key))
            last_tick = tick
        # Apply event
        if typ == 1:
            active.add(note)
        else:
            active.discard(note)

    # Merge consecutive segments with same label and ignore zero-length
    merged = []
    for s in segments:
        if s[0] == s[1]:
            continue
        if merged and merged[-1][2] == s[2] and merged[-1][3] == s[3] and merged[-1][1] == s[0]:
            merged[-1] = (merged[-1][0], s[1], s[2], s[3])
        else:
            merged.append(s)

    # Convert to seconds and write file
    from pathlib import Path
    out = output_txt_path or str(Path(midi_file_path).with_suffix('')) + '_chords.txt'
    with open(out, 'w', encoding='utf-8') as f:
        # First line: key and mode
        f.write(f"Key: {key_name} {mode}\n")
        for start_tick, end_tick, label, out_of_key in merged:
            start_sec = _ticks_to_seconds(start_tick, tempo_map, tpq)
            end_sec = _ticks_to_seconds(end_tick, tempo_map, tpq)
            norm = _normalize_symbol_to_tone_mode(label)
            suffix = " (out_of_key)" if out_of_key and norm != 'N' else ""
            f.write(f"{start_sec:.3f} {end_sec:.3f} {norm}{suffix}\n")
    return out


_PITCH_CLASS_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Full vocabulary of qualities supported by btc_chords.Chords._shorthands.
# Each entry maps a template (intervals above the root, in semitones) to the
# quality suffix written after the root; an empty suffix means bare major
# (btc parses "A" == "A:maj").
_CHORD_TEMPLATES = [
    # 6-note extensions
    (frozenset({0, 2, 4, 5, 7, 10}), ':11'),
    (frozenset({0, 2, 3, 5, 7, 10}), ':min11'),
    # 5-note extensions
    (frozenset({0, 2, 4, 7, 10}),    ':9'),
    (frozenset({0, 2, 4, 7, 11}),    ':maj9'),
    (frozenset({0, 2, 3, 7, 10}),    ':min9'),
    (frozenset({0, 4, 7, 9, 10}),    ':13'),
    (frozenset({0, 4, 7, 9, 11}),    ':maj13'),
    (frozenset({0, 3, 7, 9, 10}),    ':min13'),
    # 4-note tetrachords
    (frozenset({0, 4, 7, 11}),       ':maj7'),
    (frozenset({0, 3, 7, 10}),       ':min7'),
    (frozenset({0, 4, 7, 10}),       ':7'),
    (frozenset({0, 3, 6, 9}),        ':dim7'),
    (frozenset({0, 3, 6, 10}),       ':hdim7'),
    (frozenset({0, 3, 7, 11}),       ':minmaj7'),
    (frozenset({0, 4, 7, 9}),        ':maj6'),
    (frozenset({0, 3, 7, 9}),        ':min6'),
    (frozenset({0, 2, 4, 7}),        ':add9'),
    (frozenset({0, 2, 7, 10}),       ':7sus2'),
    (frozenset({0, 5, 7, 10}),       ':7sus4'),
    # 3-note triads
    (frozenset({0, 4, 7}),           ''),           # major (bare root)
    (frozenset({0, 3, 7}),           ':min'),
    (frozenset({0, 3, 6}),           ':dim'),
    (frozenset({0, 4, 8}),           ':aug'),
    (frozenset({0, 2, 7}),           ':sus2'),
    (frozenset({0, 5, 7}),           ':sus4'),
    # 2-note intervals
    (frozenset({0, 7}),              ':5'),   # power chord
    (frozenset({0, 5}),              ':4'),   # perfect 4th
    (frozenset({0, 9}),              ':6'),   # major 6th
    # 1-note
    (frozenset({0}),                 ':1'),
]


def _identify_chord_from_pitches(pitches):
    """Identify a BTC-format chord label from a list of MIDI pitch numbers.

    Always returns a label that btc_chords.Chords.chord() can parse, chosen by
    scoring every (root, quality) pair against the input pitch-class set:
      1. minimise unexplained input tones (|input - template|)
      2. minimise implied template tones absent from the voicing
      3. prefer roots that are actually voiced
      4. prefer root == bass (so we only add /bass when musically warranted)
      5. tie-break by larger template (more specific label)
    """
    if not pitches:
        return 'N'

    input_pcs = frozenset(p % 12 for p in pitches)
    bass_pc = min(pitches) % 12

    best_score = None
    best_label = None
    for root_pc in range(12):
        intervals = frozenset((pc - root_pc) % 12 for pc in input_pcs)
        bass_interval = (bass_pc - root_pc) % 12
        for template, quality in _CHORD_TEMPLATES:
            missing = len(intervals - template)
            implied = len(template - intervals)
            score = (
                -missing,
                -implied,
                1 if root_pc in input_pcs else 0,
                1 if root_pc == bass_pc else 0,
                # Slash note should be a real chord tone; penalise otherwise.
                1 if bass_interval in template else 0,
                len(template),
            )
            if best_score is None or score > best_score:
                best_score = score
                root_name = _PITCH_CLASS_NAMES[root_pc]
                if root_pc == bass_pc:
                    best_label = f"{root_name}{quality}"
                else:
                    bass_name = _PITCH_CLASS_NAMES[bass_pc]
                    best_label = f"{root_name}{quality}/{bass_name}"

    return best_label


def _extract_chords_from_track(chord_track, tpq):
    """Extract per-beat chord labels from a miditoolkit Instrument (Chords track).

    Returns a list of chord label strings, one per quarter-note beat,
    covering the full duration of the chord track.
    """
    from collections import defaultdict

    # Group notes by onset tick
    groups = defaultdict(list)
    max_end_tick = 0
    for note in chord_track.notes:
        groups[note.start].append(note)
        max_end_tick = max(max_end_tick, note.end)

    # Build (start_beat, end_beat, label) spans
    sorted_onsets = sorted(groups.keys())
    chord_spans = []
    for onset in sorted_onsets:
        notes = groups[onset]
        end_tick = max(n.end for n in notes)
        start_beat = round(onset / tpq)
        end_beat = round(end_tick / tpq)
        pitches = [n.pitch for n in notes]
        label = _identify_chord_from_pitches(pitches)
        chord_spans.append((start_beat, end_beat, label))

    total_beats = round(max_end_tick / tpq)
    if total_beats <= 0:
        return []

    chords_per_beat = ['N'] * total_beats
    for start_beat, end_beat, label in chord_spans:
        # Sub-beat decorations (end_beat <= start_beat after rounding) would
        # otherwise be dropped entirely — guarantee each onset labels at least
        # the beat it starts on.
        end_beat = max(end_beat, start_beat + 1)
        for b in range(start_beat, min(end_beat, total_beats)):
            chords_per_beat[b] = label

    return chords_per_beat


def export_chords_txt_chorder(midi_file_path, output_txt_path=None, beats=True, beat_times=None, beat_subdivision=1):
    """
    Export chord labels from a MIDI file to a text file.

    If the MIDI contains a 'Chords' track, chord labels are extracted directly
    from that track (faithful to the generated harmonisation).  Otherwise falls
    back to chorder's Dechorder (beat-level re-detection from all notes).

    Args:
        midi_file_path: Path to MIDI file
        output_txt_path: Output txt path (optional)
        beats: Whether to use beat-level detection
        beat_times: Optional array of actual beat times (from SingNet) to remap chord times.
                   If provided, chord times will be aligned to these actual beat positions
                   instead of using the fixed MIDI tempo.
        beat_subdivision: What subdivision the beat_times represent (default 1 = quarter notes)
                         - 1 = quarter notes (beats)
                         - 2 = 8th notes (each chord spans 2 beat_times)
                         - 4 = 16th notes (each chord spans 4 beat_times)
    """
    from miditoolkit.midi import parser
    midi = parser.MidiFile(midi_file_path)
    tpq = midi.ticks_per_beat

    # Try to extract chords directly from the Chords track
    chord_track = None
    for inst in midi.instruments:
        if 'chord' in inst.name.lower() and inst.notes:
            chord_track = inst
            break

    if chord_track is not None:
        chords = _extract_chords_from_track(chord_track, tpq)
        print(f"Extracted {len(chords)} beat-level chords from '{chord_track.name}' track")

        # Chorderator anchors chord bar 1 beat 1 to the first melody note's
        # position (== note_shift), NOT to the MIDI bar boundary. When beat_times
        # is supplied we want audio bar 1 beat 1 (== beat_times[0]) to line up
        # with the first *real* chord, so drop the leading N-beats produced by
        # that anchor offset.
        if beat_times is not None:
            first_chord_tick = min(n.start for n in chord_track.notes)
            lead_beats = int(round(first_chord_tick / tpq))
            # Only trim if those beats really are N (safety against weird inputs)
            if lead_beats > 0 and all(c == 'N' for c in chords[:lead_beats]):
                chords = chords[lead_beats:]
                print(f"Dropped {lead_beats} leading N-beats to align first chord to beat_times[0]")
    else:
        from chorder import Dechorder
        chords_raw = Dechorder.dechord(midi)
        chords = []
        for symbol in chords_raw:
            label = 'N'
            if symbol:
                try:
                    label = str(symbol)
                except Exception:
                    label = 'N'
            chords.append(_normalize_symbol_to_tone_mode(label))
        print(f"No Chords track found; fell back to Dechorder ({len(chords)} beats)")

    # Build tempo map for tick→seconds conversion (used when beat_times is None)
    tempo_changes = sorted(midi.tempo_changes, key=lambda t: t.time)
    try:
        _score_tmp = Score(midi_file_path, ttype='tick')
        qpm = float(getattr(_score_tmp, 'tempo', 0.0)) if hasattr(_score_tmp, 'tempo') else 0.0
        if qpm and qpm > 0:
            tempo_changes = [type('T', (), {'time': 0, 'tempo': qpm})()]
    except Exception:
        pass
    if not tempo_changes:
        tempo_changes = [type('T', (), {'time': 0, 'tempo': 120.0})()]

    def ticks_to_seconds_bpm(ticks: int) -> float:
        seconds = 0.0
        last_tick = 0
        last_bpm = tempo_changes[0].tempo
        for i in range(1, len(tempo_changes)):
            t = tempo_changes[i]
            if ticks <= t.time:
                seg_ticks = ticks - last_tick
                seconds += (seg_ticks / tpq) * (60.0 / last_bpm)
                return seconds
            seg_ticks = t.time - last_tick
            seconds += (seg_ticks / tpq) * (60.0 / last_bpm)
            last_tick = t.time
            last_bpm = t.tempo
        seg_ticks = ticks - last_tick
        seconds += (seg_ticks / tpq) * (60.0 / last_bpm)
        return seconds

    def get_beat_time(beat_idx):
        """Get actual beat time from beat_times array, or extrapolate if out of range."""
        if beat_times is None:
            return ticks_to_seconds_bpm(beat_idx * tpq)

        actual_idx = beat_idx * beat_subdivision

        if actual_idx < len(beat_times):
            return beat_times[actual_idx]
        else:
            if len(beat_times) >= 2:
                last_intervals = []
                for i in range(max(0, len(beat_times)-5), len(beat_times)-1):
                    last_intervals.append(beat_times[i+1] - beat_times[i])
                avg_interval = sum(last_intervals) / len(last_intervals) if last_intervals else 0.5
                return beat_times[-1] + (actual_idx - len(beat_times) + 1) * avg_interval
            else:
                return ticks_to_seconds_bpm(beat_idx * tpq)

    # Write
    from pathlib import Path
    out = output_txt_path or str(Path(midi_file_path).with_suffix('')) + '_chords_chorder.txt'
    with open(out, 'w', encoding='utf-8') as f:
        for i, label in enumerate(chords):
            start_sec = get_beat_time(i)
            end_sec = get_beat_time(i + 1)
            f.write(f"{start_sec:.3f} {end_sec:.3f} {label}\n")

    if beat_times is not None:
        subdivision_name = {1: "quarter notes", 2: "8th notes", 4: "16th notes"}.get(beat_subdivision, f"1/{beat_subdivision} notes")
        print(f"Chord times remapped to {len(beat_times)} beat positions (subdivision: {subdivision_name})")

    return out


def fill_none_chords_in_txt(input_txt_path, output_txt_path=None):
    """
    Fill 'None' or 'N' chords in a chord txt file by extending the previous chord.

    Mid-song None rows are filled by carrying the previous valid chord forward.
    Leading None rows (before any valid chord exists) are backfilled from the
    first valid chord found in the file. If the first emitted chord starts after
    time 0, extend that first valid chord back to 0.0 so the exported text/BTC
    covers leading silence or pickup audio as well.

    Args:
        input_txt_path: Path to input chord txt file
        output_txt_path: Path to output file (optional, defaults to same as input with _filled suffix)

    Returns:
        str: Path to output file
    """
    from pathlib import Path

    lines = []
    with open(input_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 3:
                    start, end, chord = parts[0], parts[1], parts[2]
                    lines.append([float(start), float(end), chord])

    # Forward fill: replace each None with the most recent valid chord.
    last_valid_chord = None
    none_count = 0
    for i, (_s, _e, chord) in enumerate(lines):
        if chord.lower() in ('none', 'n'):
            if last_valid_chord is not None:
                lines[i][2] = last_valid_chord
                none_count += 1
        else:
            last_valid_chord = chord

    # Backfill any leading None rows that the forward pass couldn't reach
    # (those before the first valid chord).
    first_valid_idx = None
    for i, (_s, _e, chord) in enumerate(lines):
        if chord.lower() not in ('none', 'n'):
            first_valid_idx = i
            break

    if first_valid_idx is not None and first_valid_idx > 0:
        first_valid_chord = lines[first_valid_idx][2]
        for i in range(first_valid_idx):
            if lines[i][2].lower() in ('none', 'n'):
                lines[i][2] = first_valid_chord
                none_count += 1

    # If export was downbeat-anchored, there may be no explicit rows before the
    # first valid chord at all. Extend that first chord back to 0.0 so the
    # downstream BTC/chord conditioning covers the song from the beginning.
    leading_gap_extended = 0.0
    if lines and lines[0][2].lower() not in ('none', 'n') and lines[0][0] > 0:
        leading_gap_extended = lines[0][0]
        lines[0][0] = 0.0

    # Write output
    if output_txt_path is None:
        base = str(Path(input_txt_path).with_suffix(''))
        output_txt_path = base + '_filled.txt'

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for start, end, chord in lines:
            f.write(f"{start:.3f} {end:.3f} {chord}\n")

    if none_count > 0:
        print(f"Filled {none_count} None chords")
    if leading_gap_extended > 0:
        print(f"Extended first chord back to 0.000s (covered leading gap of {leading_gap_extended:.3f}s)")

    return output_txt_path


# --- Chord symbol normalization ---
def _normalize_symbol_to_tone_mode(symbol: str) -> str:
    """Normalize chord symbols to 'Tone[:quality]' style, e.g., CM->C, Am->A:min, Gm7->G:min7.
    Keeps 'N' as is. Accepts inputs from chorder like 'CM','Am','Dm','G','Bb','Gm7'.
    """
    if not symbol:
        return 'N'
    s = symbol.strip()
    if s.upper() == 'N':
        return 'N'
    import re
    m = re.match(r'^([A-G](?:#|b)?)(.*)$', s)
    if not m:
        return s
    root, qual = m.group(1), m.group(2)
    qual = qual.strip()

    # Split slash bass off before quality matching so it survives normalization.
    # Chorder uses formats like 'C#M/G#', 'F#M7/C#', 'Ebm7/Bb'.
    bass = ''
    if '/' in qual:
        qual, _b = qual.split('/', 1)
        bass = '/' + _b

    # IMPORTANT: chorder distinguishes major from minor by case — 'M' = major,
    # 'm' = minor. Lowercasing the whole quality conflates them, so the m/M
    # branches below must be case-sensitive on `q`.
    q = qual
    q_low = q.lower()

    # Major (chorder uses 'M'); bare major has no quality suffix
    if q in ('', 'M') or q_low == 'maj':
        return root + bass
    if q == 'M7' or q_low == 'maj7':
        return f"{root}:maj7{bass}"
    if q == 'M9' or q_low == 'maj9':
        return f"{root}:maj9{bass}"

    # Minor (chorder uses lowercase 'm')
    if q == 'm' or q_low == 'min':
        return f"{root}:min{bass}"
    if q == 'm7' or q_low == 'min7':
        return f"{root}:min7{bass}"
    if q == 'm9' or q_low == 'min9':
        return f"{root}:min9{bass}"

    # Qualities with no m/M ambiguity — case-insensitive is fine
    if q_low in ('7',):
        return f"{root}:7{bass}"
    if q_low in ('dim', 'o'):
        return f"{root}:dim{bass}"
    if q_low in ('dim7', 'o7'):
        return f"{root}:dim7{bass}"
    if q_low in ('aug', '+'):
        return f"{root}:aug{bass}"
    if q_low in ('sus2',):
        return f"{root}:sus2{bass}"
    if q_low in ('sus4',):
        return f"{root}:sus4{bass}"

    # Prefix fallbacks — case-sensitive on the m-branches only
    if q.startswith('m7'):
        return f"{root}:min7{bass}"
    if q.startswith('m'):
        return f"{root}:min{bass}"
    if q_low.startswith('maj7'):
        return f"{root}:maj7{bass}"
    if q_low.startswith('maj9'):
        return f"{root}:maj9{bass}"
    if q_low.startswith('dim7'):
        return f"{root}:dim7{bass}"
    if q_low.startswith('dim'):
        return f"{root}:dim{bass}"
    if q_low.startswith('aug'):
        return f"{root}:aug{bass}"
    if q_low.startswith('sus2'):
        return f"{root}:sus2{bass}"
    if q_low.startswith('sus4'):
        return f"{root}:sus4{bass}"
    if q_low.startswith('7'):
        return f"{root}:7{bass}"

    # Fallback: return root + cleaned qual if any
    return root + bass if not q else f"{root}:{q}{bass}"


# --- Tempo utilities ---
def get_initial_bpm(midi_path: str) -> float:
    """Read initial tempo (BPM) from a MIDI file. Fallback to 120 if missing."""
    try:
        # Prefer miditoolkit
        from miditoolkit.midi import parser
        mf = parser.MidiFile(midi_path)
        if mf.tempo_changes:
            return float(mf.tempo_changes[0].tempo)
    except Exception:
        pass
    try:
        # Fallback mido
        import mido as _m
        f = _m.MidiFile(midi_path)
        abs_t = 0
        for msg in _m.merge_tracks(f.tracks):
            abs_t += msg.time
            if msg.type == 'set_tempo':
                return 60_000_000.0 / msg.tempo
    except Exception:
        pass
    return 120.0


def set_midi_global_bpm(target_midi_path: str, bpm: float, output_path: Optional[str] = None) -> str:
    """Set/override global tempo at tick 0 for a MIDI file, preserving other events.
    Writes to output_path (or in-place if None) and returns the path written."""
    import mido as _m
    mid = _m.MidiFile(target_midi_path)
    tempo_us = int(round(60_000_000.0 / bpm))
    # Ensure tempo at the very beginning of track 0
    for ti, tr in enumerate(mid.tracks):
        # Insert set_tempo at time 0 at the very start
        # Keep existing delta times by pushing current first event after tempo if needed
        if ti == 0:
            # If there's already a tempo at start, replace it
            if tr and getattr(tr[0], 'type', None) == 'set_tempo' and tr[0].time == 0:
                tr[0] = _m.MetaMessage('set_tempo', tempo=tempo_us, time=0)
            else:
                tr.insert(0, _m.MetaMessage('set_tempo', tempo=tempo_us, time=0))
            break
    out = output_path or target_midi_path
    mid.save(out)
    return out


def sync_output_tempo_with_input(input_midi_path: str, output_midi_paths: list[str]) -> list[str]:
    """Make all output MIDI files use the same tempo as the input MIDI.
    Returns the list of written paths (same as inputs)."""
    bpm = get_initial_bpm(input_midi_path)
    written = []
    for p in output_midi_paths:
        written.append(set_midi_global_bpm(p, bpm))
    return written

def fill_empty_bars_with_chords(input_melody_path, midi_file_path, empty_bars, output_path=None):
    """
    Fill empty bars at the beginning with chords from the chord generation using symusic.
    
    This function does two things:
    1. Aligns chord timing with the original melody by converting TPQ
    2. Copies chord pattern from the beginning of generated chords to fill empty bars
       (i.e., if chords start at bar N, copy the first few bars of chords to bar 0)
    
    Args:
        input_melody_path: Path to the original melody MIDI file
        midi_file_path: Path to the generated chord MIDI file
        empty_bars: Number of empty bars at the beginning (before melody starts)
        output_path: Optional output path, defaults to input path with '_filled' suffix
    
    Returns:
        str: Path to the output file
    """
    from symusic import Note
    
    # Load MIDI files
    chord_score = Score(midi_file_path, ttype="tick")
    original_score = Score(input_melody_path, ttype="tick")
    
    # Get TPQ values
    original_tpq = original_score.ticks_per_quarter
    chord_tpq = chord_score.ticks_per_quarter
    tpq_ratio = original_tpq / chord_tpq
    
    print(f"=== Fill Empty Bars with Chords ===")
    print(f"Original melody TPQ: {original_tpq}")
    print(f"Chord gen TPQ: {chord_tpq}")
    print(f"TPQ ratio: {tpq_ratio:.4f}")
    
    if len(chord_score.tracks) < 2:
        print("Warning: MIDI file doesn't have both piano and chord tracks")
        return midi_file_path
    
    # Track indices (assuming standard AccoMontage output)
    piano_track_idx = 0
    chord_track_idx = 1
    
    # Get chord track
    chord_track_obj = chord_score.tracks[chord_track_idx]
    if not chord_track_obj.notes:
        print("Warning: No chord notes found")
        return midi_file_path
    
    # Calculate timing constants
    ticks_per_bar_chord = chord_tpq * 4  # 4 beats per bar in 4/4 time
    ticks_per_bar_original = original_tpq * 4
    
    # Find where chords actually start
    first_chord_tick = chord_track_obj.notes[0].start
    first_chord_bar = first_chord_tick / ticks_per_bar_chord
    
    print(f"Chords start at tick: {first_chord_tick} (bar {first_chord_bar:.2f})")
    print(f"Empty bars to fill: {empty_bars}")
    
    # Create new score with original TPQ
    new_score = Score(ttype="tick")
    new_score.ticks_per_quarter = original_tpq
    new_score.tempos = original_score.tempos
    
    from symusic import Track
    new_chord_track = Track(name="Chords")
    
    # Helper function to convert tick from chord TPQ to original TPQ
    def convert_tick(tick_in_chord_tpq):
        return int(tick_in_chord_tpq * tpq_ratio)
    
    # If the generated MIDI was created on a compressed detected-only timeline,
    # chords can already start at bar 0 even though the full-song timeline has
    # leading empty bars. In that case we need to prepend bars by shifting the
    # generated content to the right, then copy the opening chord pattern back
    # to bar 0 to fill the intro.
    first_chord_bar_int = int(first_chord_bar)
    shift_bars = max(0, int(empty_bars) - first_chord_bar_int)
    shift_ticks_original = shift_bars * ticks_per_bar_original

    # STEP 1: Fill the leading region before the "real" chord entrance.
    fill_bars = max(int(empty_bars), first_chord_bar_int)
    if fill_bars > 0:
        pattern_duration_chord = fill_bars * ticks_per_bar_chord

        # Collect notes from the first section of chords.
        pattern_notes = []
        for note in chord_track_obj.notes:
            relative_start = note.start - first_chord_tick
            if relative_start < pattern_duration_chord:
                pattern_notes.append(note)
            else:
                break

        print(
            f"Prepending {shift_bars} bars and copying {len(pattern_notes)} chord notes "
            f"to cover {fill_bars} leading bars"
        )

        for note in pattern_notes:
            new_start = convert_tick(note.start - first_chord_tick)
            new_duration = convert_tick(note.end - note.start)
            new_note = Note(
                time=new_start,
                duration=new_duration,
                pitch=note.pitch,
                velocity=note.velocity
            )
            new_chord_track.notes.append(new_note)
    
    # STEP 2: Add all original chord notes with converted timing
    for note in chord_track_obj.notes:
        new_start = convert_tick(note.start) + shift_ticks_original
        new_duration = convert_tick(note.end - note.start)
        new_note = Note(
            time=new_start,
            duration=new_duration,
            pitch=note.pitch,
            velocity=note.velocity
        )
        new_chord_track.notes.append(new_note)
    
    # Sort and remove duplicates (notes at same position with same pitch)
    new_chord_track.notes.sort(key=lambda n: (n.time, n.pitch))
    
    # Remove exact duplicates
    unique_notes = []
    for note in new_chord_track.notes:
        if not unique_notes or (note.time != unique_notes[-1].time or 
                                note.pitch != unique_notes[-1].pitch):
            unique_notes.append(note)
    new_chord_track.notes = unique_notes
    
    # STEP 3: Add melody track from chord file.
    # Anchor it to the INPUT melody's true first-note position rather than
    # reusing shift_ticks_original. Chorderator re-emits its melody track at the
    # melody's original pickup offset, so adding shift_ticks_original (sized for
    # re-aligning the bar-0 chord track) would push the melody an extra
    # `empty_bars` bars late whenever the pickup is a whole bar. Computing the
    # shift from the input melody's onset is correct whether chorderator kept
    # the pickup (shift -> 0) or stripped it to bar 0 (shift -> pickup bars).
    new_melody_track = Track(name="Melody")
    melody_track_from_chord = chord_score.tracks[piano_track_idx]
    if melody_track_from_chord.notes and original_score.tracks and original_score.tracks[piano_track_idx].notes:
        desired_first = min(n.start for n in original_score.tracks[piano_track_idx].notes)
        current_first = convert_tick(min(n.start for n in melody_track_from_chord.notes))
        melody_shift = desired_first - current_first
    else:
        melody_shift = shift_ticks_original
    for note in melody_track_from_chord.notes:
        new_start = convert_tick(note.start) + melody_shift
        new_duration = convert_tick(note.end - note.start)
        new_note = Note(
            time=new_start,
            duration=new_duration,
            pitch=note.pitch,
            velocity=note.velocity
        )
        new_melody_track.notes.append(new_note)
    
    new_melody_track.notes.sort(key=lambda n: n.time)
    
    # Add tracks to new score
    new_score.tracks.append(new_melody_track)
    new_score.tracks.append(new_chord_track)
    
    # Determine output path
    if output_path is None:
        base_path = os.path.splitext(midi_file_path)[0]
        output_path = f"{base_path}_filled_empty_bars.mid"
    
    # Save the modified MIDI file
    new_score.dump_midi(output_path)
    
    print(f"\n=== Output Summary ===")
    print(f"Output TPQ: {new_score.ticks_per_quarter}")
    print(f"Melody track notes: {len(new_melody_track.notes)}")
    print(f"Chord track notes: {len(new_chord_track.notes)}")
    if new_chord_track.notes:
        print(f"First chord now at tick: {new_chord_track.notes[0].time}")
        print(f"Last chord at tick: {new_chord_track.notes[-1].time}")
    print(f"Output saved to: {output_path}")

    return output_path


def fill_internal_empty_bars(midi_path, chord_track_name="Chords"):
    """
    Fill mid-song empty chord bars by replaying the surrounding progression.
    For a gap of K bars, copy the K bars immediately preceding the gap (so
    the listener perceives a phrase repeat rather than a held drone). If
    there aren't K preceding non-empty bars, fall back to the K bars after
    the gap; if the gap is longer than the available source, cycle through it.
    """
    from symusic import Note
    score = Score(midi_path, ttype="tick")
    tpq = score.ticks_per_quarter
    bar_ticks = tpq * 4

    chord_track = next((t for t in score.tracks if t.name == chord_track_name), None)
    if chord_track is None or not chord_track.notes:
        return midi_path

    notes = list(chord_track.notes)
    # Chorderator sometimes emits downbeat events 1 tick early (e.g. tick 3839
    # instead of 3840). Bucket with a 32nd-note tolerance so those events land
    # in the bar they were musically intended for.
    tol = tpq // 8

    def bar_of(tick):
        return (tick + tol) // bar_ticks

    first_bar = min(bar_of(n.time) for n in notes)
    last_bar  = max(bar_of(n.time) for n in notes)
    by_bar = {}
    for n in notes:
        by_bar.setdefault(bar_of(n.time), []).append(n)

    nonempty_bars = sorted(by_bar.keys())
    nonempty_set = set(nonempty_bars)

    # Group consecutive empty bars into (start, end_inclusive) gaps.
    gaps = []
    bar = first_bar
    while bar <= last_bar:
        if bar in nonempty_set:
            bar += 1
            continue
        start = bar
        while bar <= last_bar and bar not in nonempty_set:
            bar += 1
        gaps.append((start, bar - 1))

    added = 0
    filled_summary = []
    for gap_start, gap_end in gaps:
        gap_len = gap_end - gap_start + 1
        # Source: the K=gap_len most recent contiguous non-empty bars BEFORE the gap
        src_bars = [b for b in nonempty_bars if b < gap_start][-gap_len:]
        if not src_bars:
            # Fall back to bars AFTER the gap
            src_bars = [b for b in nonempty_bars if b > gap_end][:gap_len]
        if not src_bars:
            continue  # entire chord track is empty (shouldn't happen here)

        for i, dst_bar in enumerate(range(gap_start, gap_end + 1)):
            src_bar = src_bars[i % len(src_bars)]
            shift = (dst_bar - src_bar) * bar_ticks
            for n in by_bar[src_bar]:
                chord_track.notes.append(Note(
                    time=n.time + shift,
                    duration=n.duration,
                    pitch=n.pitch,
                    velocity=n.velocity,
                ))
                added += 1
        filled_summary.append(f"bars {gap_start}-{gap_end}←{src_bars}")

    if added:
        chord_track.notes.sort(key=lambda n: (n.time, n.pitch))
        score.dump_midi(midi_path)
        print(f"Filled {len(gaps)} internal chord gaps ({added} notes): "
              f"{filled_summary[:6]}{'...' if len(filled_summary) > 6 else ''}")
    return midi_path


def cover_pickup_melody_with_chord(midi_path, melody_track_name="Melody",
                                   chord_track_name="Chords"):
    """
    Extend the first chord event backward so the melody pickup has chord
    coverage in the MIDI representation. Text/BTC export handles the leading
    silence separately; this helper only patches the chord track inside the
    MIDI so it reads better musically/visually.
    """
    from symusic import Note
    score = Score(midi_path, ttype="tick")

    chord_track = next((t for t in score.tracks if t.name == chord_track_name), None)
    melody_track = next((t for t in score.tracks if t.name == melody_track_name), None)
    if chord_track is None or not chord_track.notes:
        return midi_path
    if melody_track is None or not melody_track.notes:
        return midi_path

    first_chord_tick = min(n.time for n in chord_track.notes)
    first_melody_tick = min(n.time for n in melody_track.notes)
    if first_melody_tick >= first_chord_tick:
        return midi_path

    pickup_duration = first_chord_tick - first_melody_tick
    first_chord_notes = [n for n in chord_track.notes if n.time == first_chord_tick]
    for n in first_chord_notes:
        chord_track.notes.append(Note(
            time=first_melody_tick,
            duration=pickup_duration,
            pitch=n.pitch,
            velocity=n.velocity,
        ))
    chord_track.notes.sort(key=lambda n: (n.time, n.pitch))
    score.dump_midi(midi_path)
    print(f"Added pickup chord ({len(first_chord_notes)} pitches) over "
          f"ticks [{first_melody_tick}, {first_chord_tick}] "
          f"({pickup_duration/score.ticks_per_quarter:.2f} beats)")
    return midi_path


def scale_midi_ticks(src_path, dst_path, scale):
    """
    Scale every time-indexed MIDI event by `scale`, keeping tpq unchanged.

    Used to emulate "N chords per bar" for chorderator: scaling the melody by
    2x makes chorderator see each real half-bar as a full bar, so it picks a
    distinct chord for each half-bar based on that half-bar's melody content.
    Apply the inverse scale (e.g. 0.5) to the generated chord MIDI to restore
    real-time alignment.
    """
    score = Score(src_path, ttype="tick")

    def _s(t):
        return int(round(t * scale))

    for track in score.tracks:
        for n in track.notes:
            n.time = _s(n.time)
            n.duration = max(1, _s(n.duration))
        for c in track.controls:
            c.time = _s(c.time)
        for p in track.pedals:
            p.time = _s(p.time)
            p.duration = max(1, _s(p.duration))
        for pb in track.pitch_bends:
            pb.time = _s(pb.time)
    # Scale tempo BPM too: stretching ticks by `scale` with unchanged BPM would
    # stretch wall-clock by `scale`, confusing downstream tools (chorderator)
    # that convert between tick and seconds. Keep wall-clock constant.
    for tempo in score.tempos:
        tempo.time = _s(tempo.time)
        tempo.qpm = tempo.qpm * scale
    for ts in score.time_signatures:
        ts.time = _s(ts.time)
    for ks in score.key_signatures:
        ks.time = _s(ks.time)
    for m in score.markers:
        m.time = _s(m.time)

    score.dump_midi(dst_path)
    return dst_path


def align_chord_track_to_bar(midi_path, beats_per_bar=4, chord_track_name="Chords",
                             chord_resolution_beats=None):
    """
    Snap every chord onset to the nearest grid point defined by
    `chord_resolution_beats` (default == beats_per_bar, i.e., bar grid).

    Chorderator places its chord-bar-1-beat-1 at the melody's first note, so
    chord changes can sit off the audio downbeat by a fractional bar. Earlier
    steps like fill_empty_bars_with_chords can also leave onsets on a different
    grid (real bars) than chorderator's (note_shift-offset). Per-onset snapping
    reconciles both — every onset lands on the nearest grid point regardless of
    which pipeline step produced it. When several original onsets collapse into
    the same snapped bucket, keep the latest onset's pitch set (usually the
    chorderator-native chord rather than a prepended fill) but extend that
    snapped chord to the next snapped bucket. This prevents short ornamental
    re-voicings from overwriting a bar-level chord and leaving audible gaps.

    BTC export is unaffected (must be called after BTC is written).
    """
    from symusic import Note
    from collections import defaultdict

    score = Score(midi_path, ttype="tick")
    tpq = score.ticks_per_quarter
    resolution = chord_resolution_beats or beats_per_bar
    grid_ticks = tpq * resolution

    chord_track = next((t for t in score.tracks if t.name == chord_track_name), None)
    if chord_track is None or not chord_track.notes:
        return midi_path

    def snap(t):
        return int(round(t / grid_ticks) * grid_ticks)

    # Group by original onset, then bucket those onsets by the snapped grid.
    # If several original onsets land in the same bucket, keep the latest
    # group's pitches (preserving the existing chord label behavior) but stretch
    # that snapped chord to the next snapped bucket so short decorations do not
    # create bar-long gaps.
    by_onset = defaultdict(list)
    for n in chord_track.notes:
        by_onset[n.time].append(n)

    snapped = defaultdict(list)
    for onset in sorted(by_onset):
        snapped[snap(onset)].append((onset, by_onset[onset]))

    snapped_onsets = sorted(snapped.keys())

    new_notes = []
    snap_count = 0
    extended_count = 0
    merged_buckets = 0
    for i, new_onset in enumerate(snapped_onsets):
        entries = snapped[new_onset]
        if len(entries) > 1:
            merged_buckets += 1

        # Keep the latest onset's pitches inside this snapped bucket.
        original_onset, notes = entries[-1]
        next_onset = snapped_onsets[i + 1] if i + 1 < len(snapped_onsets) else None
        original_end = max(n.time + n.duration for n in notes)
        target_end = next_onset if next_onset is not None else max(original_end, new_onset + grid_ticks)
        if target_end <= new_onset:
            target_end = max(original_end, new_onset + grid_ticks)
        snapped_duration = max(1, target_end - new_onset)

        seen_pitches = set()
        for n in sorted(notes, key=lambda note: note.pitch):
            if n.pitch in seen_pitches:
                continue
            seen_pitches.add(n.pitch)
            if n.time != new_onset:
                snap_count += 1
            if n.duration != snapped_duration:
                extended_count += 1
            new_notes.append(Note(time=new_onset, duration=snapped_duration,
                                  pitch=n.pitch, velocity=n.velocity))
    new_notes.sort(key=lambda n: (n.time, n.pitch))
    chord_track.notes = new_notes

    if snap_count or extended_count or merged_buckets:
        print(f"Snapped {snap_count} chord notes to {resolution}-beat grid "
              f"(total chord onsets: {len(snapped_onsets)}, "
              f"merged buckets: {merged_buckets}, extended notes: {extended_count})")
    score.dump_midi(midi_path)
    return midi_path


def estimate_tempo_from_notes(input_melody_path):
    """
    Estimate the correct tempo by analyzing note positions.
    
    IMPORTANT: This function assumes the MIDI file has notes positioned based on
    actual audio timing, and we need to find the tempo that makes the notes
    align to a musical grid.
    
    The approach:
    1. Find the most common inter-onset interval (IOI)
    2. Calculate what tempo would make this IOI a 1/16, 1/8, or 1/4 note
    3. Pick the tempo in a reasonable range (60-200 BPM)
    
    Args:
        input_melody_path: Path to the MIDI file
    
    Returns:
        float: Estimated BPM
    """
    from collections import Counter
    
    score = Score(input_melody_path, ttype="tick")
    tpq = score.ticks_per_quarter
    
    # Collect all note onsets
    all_notes = []
    for track in score.tracks:
        all_notes.extend(track.notes)
    
    if not all_notes:
        print("No notes found, using default 120 BPM")
        return 120.0
    
    # Get note onset positions
    onsets = sorted(set(n.start for n in all_notes))
    
    if len(onsets) < 2:
        print("Less than 2 onsets, using default 120 BPM")
        return 120.0
    
    # Calculate inter-onset intervals (IOIs)
    iois = [onsets[i+1] - onsets[i] for i in range(len(onsets)-1)]
    
    # Filter out very small IOIs (likely grace notes or chords)
    min_ioi = tpq / 8  # 1/32 note at 120 BPM
    iois = [ioi for ioi in iois if ioi >= min_ioi]
    
    if not iois:
        print("No valid IOIs found, using default 120 BPM")
        return 120.0
    
    # Quantize IOIs to reduce noise (to 1/16 note resolution at 120 BPM)
    quantize_unit = tpq / 4  # 1/16 note at 120 BPM = 120 ticks
    quantized_iois = [round(ioi / quantize_unit) * quantize_unit for ioi in iois]
    ioi_counts = Counter(quantized_iois)
    
    # Get the most common IOI
    most_common_ioi = ioi_counts.most_common(1)[0][0]
    print(f"Most common IOI: {most_common_ioi} ticks")
    print(f"Top 5 IOIs: {ioi_counts.most_common(5)}")
    
    if most_common_ioi <= 0:
        print("Invalid IOI, using default 120 BPM")
        return 120.0
    
    # Calculate what tempo would make this IOI equal to different note values
    # Formula: At tempo T, note duration in ticks = (60/T) * (note_fraction * 4) * tpq / 60
    #          = note_fraction * 4 * tpq * (60/T) / 60
    #          = note_fraction * 4 * tpq / T * (standard tempo / standard tempo)
    # 
    # Simplified: At 120 BPM, 1/4 note = tpq ticks, 1/8 = tpq/2, 1/16 = tpq/4
    # If actual IOI should be X note at tempo T:
    #   most_common_ioi / (tpq * note_fraction * 4) = 120 / T
    #   T = 120 * (tpq * note_fraction * 4) / most_common_ioi
    
    candidates = []
    
    # If IOI is 1/16 note: T = 120 * (tpq/4) / most_common_ioi
    tempo_if_16th = 120.0 * (tpq / 4) / most_common_ioi
    
    # If IOI is 1/8 note: T = 120 * (tpq/2) / most_common_ioi
    tempo_if_8th = 120.0 * (tpq / 2) / most_common_ioi
    
    # If IOI is 1/4 note (beat): T = 120 * tpq / most_common_ioi
    tempo_if_quarter = 120.0 * tpq / most_common_ioi
    
    print(f"Tempo candidates: 1/16={tempo_if_16th:.1f}, 1/8={tempo_if_8th:.1f}, 1/4={tempo_if_quarter:.1f} BPM")
    
    # Choose the tempo that gives a reasonable BPM (60-200 range is typical)
    for tempo in [tempo_if_16th, tempo_if_8th, tempo_if_quarter]:
        if 60 <= tempo <= 200:
            candidates.append(tempo)
    
    if not candidates:
        # Try wider range if nothing found
        for tempo in [tempo_if_16th, tempo_if_8th, tempo_if_quarter]:
            if 40 <= tempo <= 240:
                candidates.append(tempo)
    
    if candidates:
        # Prefer tempo closer to 120 (common default)
        estimated_tempo = min(candidates, key=lambda t: abs(t - 120))
        print(f"Selected tempo: {estimated_tempo:.1f} BPM (closest to 120 in valid range)")
    else:
        estimated_tempo = 120.0
        print(f"No valid tempo found, using default 120 BPM")
    
    return estimated_tempo


def reassign_global_tempo(input_melody_path, output_melody_path=None, target_bpm=None):
    """
    Reassign global tempo to a MIDI file. This is useful when the MIDI file has
    an incorrect tempo but the note positions (in ticks) are correct.
    
    The function will:
    1. If target_bpm is provided, use that tempo
    2. Otherwise, estimate tempo by analyzing note alignment to grid
    
    Args:
        input_melody_path: Path to the input MIDI file
        output_melody_path: Optional output path, defaults to overwriting input
        target_bpm: Target BPM to set. If None, will estimate from note analysis.
    
    Returns:
        tuple: (output_path, new_bpm)
    """
    from symusic import Note, Track
    import mido
    
    score = Score(input_melody_path, ttype="tick")
    tpq = score.ticks_per_quarter
    
    print(f"=== Reassigning Global Tempo ===")
    print(f"Input: {input_melody_path}")
    print(f"TPQ: {tpq}")
    
    # Get current tempo
    current_tempo = 120.0  # default
    if score.tempos:
        current_tempo = score.tempos[0].qpm
    print(f"Current tempo in file: {current_tempo} BPM")
    
    if target_bpm is not None:
        new_bpm = target_bpm
        print(f"Using provided target BPM: {new_bpm}")
    else:
        # Estimate tempo from note positions
        new_bpm = estimate_tempo_from_notes(input_melody_path)
        print(f"Estimated tempo from note analysis: {new_bpm} BPM")
    
    # Update the tempo
    if output_melody_path is None:
        output_melody_path = input_melody_path
    
    # Use mido to properly set tempo at tick 0
    midi = mido.MidiFile(input_melody_path)
    tempo_us = int(round(60_000_000.0 / new_bpm))
    
    # Find or create tempo event at the beginning
    for track in midi.tracks:
        # Remove existing tempo events and insert new one
        new_track = []
        first_tempo_set = False
        for msg in track:
            if msg.type == 'set_tempo':
                if not first_tempo_set:
                    # Replace first tempo with our new tempo
                    new_track.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=msg.time))
                    first_tempo_set = True
                # Skip other tempo events
            else:
                new_track.append(msg)
        
        if not first_tempo_set:
            # Insert tempo at the very beginning
            new_track.insert(0, mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))
        
        track[:] = new_track
        break  # Only modify first track
    
    midi.save(output_melody_path)
    
    print(f"New tempo set: {new_bpm} BPM")
    print(f"Output saved to: {output_melody_path}")
    
    return output_melody_path, new_bpm


def pad_melody_to_valid_length(input_path, output_path=None):
    """
    Pad melody to a valid length (multiple of 4 bars) by adding empty time at the end.
    This ensures the melody can be properly segmented by chorderator.
    
    Args:
        input_path: Path to the input MIDI file
        output_path: Optional output path, defaults to overwriting input
    
    Returns:
        tuple: (output_path, original_bars, padded_bars)
    """
    from symusic import Note, Track
    
    score = Score(input_path, ttype="tick")
    tpq = score.ticks_per_quarter
    ticks_per_bar = tpq * 4  # 4/4 time
    
    # Find the end of the melody
    max_tick = 0
    for track in score.tracks:
        for note in track.notes:
            if note.end > max_tick:
                max_tick = note.end
    
    # Calculate current bar count
    current_bars = max_tick / ticks_per_bar
    original_bars = int(current_bars) + (1 if current_bars % 1 > 0 else 0)
    
    # Calculate target bar count (round UP to multiple of 4 for padding)
    target_bars = get_valid_bar_count(original_bars, round_up=True)
    
    if target_bars == original_bars:
        print(f"Melody already at valid length: {original_bars} bars")
        if output_path is None:
            return input_path, original_bars, original_bars
        # Still save to output path
        score.dump_midi(output_path)
        return output_path, original_bars, original_bars
    
    # Extend the melody by adjusting the end time
    # We don't need to add notes, just ensure the MIDI length covers the target
    target_ticks = target_bars * ticks_per_bar
    
    # Add a silent note at the end to extend the MIDI length
    # (This is a common trick to ensure MIDI players/parsers recognize the full length)
    if score.tracks:
        # Add a very quiet note at pitch 0 (or use a rest)
        # Actually, we can just extend the last note slightly or add metadata
        # For simplicity, we'll just save as-is and let chorderator handle it
        pass
    
    if output_path is None:
        output_path = input_path
    
    score.dump_midi(output_path)
    
    print(f"Melody padded: {original_bars} -> {target_bars} bars")
    return output_path, original_bars, target_bars


def preprocess_melody(input_melody_path, output_melody_path, target_bpm=None, auto_estimate_tempo=True):
    """
    Preprocess the melody:
    1. Leave only one track (the one with notes)
    2. Reassign global tempo (either to target_bpm or auto-estimated)
    3. Pad melody to valid length if needed
    
    Args:
        input_melody_path: Path to the input MIDI file
        output_melody_path: Path to save the preprocessed MIDI
        target_bpm: Target BPM to set. If None and auto_estimate_tempo is True, 
                    tempo will be estimated from note positions.
        auto_estimate_tempo: If True and target_bpm is None, estimate tempo automatically.
    
    Returns:
        float: The tempo (BPM) that was set
    """
    print(f"Preprocessing melody: {input_melody_path}")
    score = Score(input_melody_path, ttype="tick")
    for i, track in enumerate(score.tracks):
        if track.notes:
            score.tracks = [track]
            break
    score.dump_midi(output_melody_path)
    
    # Reassign tempo
    if target_bpm is not None:
        # Use specified tempo
        _, new_bpm = reassign_global_tempo(output_melody_path, output_melody_path, target_bpm)
    elif auto_estimate_tempo:
        # Auto-estimate tempo from note positions
        _, new_bpm = reassign_global_tempo(output_melody_path, output_melody_path, target_bpm=None)
    else:
        # Keep original tempo
        if score.tempos:
            new_bpm = score.tempos[0].qpm
        else:
            new_bpm = 120.0
    
    # Pad melody to valid length
    pad_melody_to_valid_length(output_melody_path, output_melody_path)
    
    return new_bpm


def align_chord_gen_tpq(original_melody_path, chord_gen_path, output_path=None):
    """
    Convert chord_gen MIDI file to have the same TPQ as the original melody.
    This ensures the chord_gen file can be properly aligned with the original melody.
    
    Args:
        original_melody_path: Path to the original melody MIDI file
        chord_gen_path: Path to the chord_gen MIDI file (from chorderator)
        output_path: Optional output path, defaults to overwriting chord_gen_path
    
    Returns:
        str: Path to the output file
    """
    from symusic import Note, Track
    
    original_score = Score(original_melody_path, ttype="tick")
    chord_score = Score(chord_gen_path, ttype="tick")
    
    original_tpq = original_score.ticks_per_quarter
    chord_tpq = chord_score.ticks_per_quarter
    
    # If TPQ is already the same, no conversion needed
    if original_tpq == chord_tpq:
        print(f"TPQ already aligned ({original_tpq}), no conversion needed")
        return chord_gen_path
    
    tpq_ratio = original_tpq / chord_tpq
    
    print(f"=== Aligning chord_gen TPQ ===")
    print(f"Original melody TPQ: {original_tpq}")
    print(f"Chord gen TPQ: {chord_tpq}")
    print(f"TPQ ratio: {tpq_ratio:.4f}")
    
    # Helper function to convert tick
    def convert_tick(tick):
        return int(tick * tpq_ratio)
    
    # Create new score with original TPQ
    new_score = Score(ttype="tick")
    new_score.ticks_per_quarter = original_tpq
    new_score.tempos = original_score.tempos
    
    # Convert all tracks
    for track in chord_score.tracks:
        new_track = Track(name=track.name)
        for note in track.notes:
            new_start = convert_tick(note.start)
            new_end = convert_tick(note.end)
            new_note = Note(
                time=new_start,
                duration=new_end - new_start,
                pitch=note.pitch,
                velocity=note.velocity
            )
            new_track.notes.append(new_note)
        new_track.notes.sort(key=lambda n: n.time)
        new_score.tracks.append(new_track)
    
    # Determine output path
    if output_path is None:
        output_path = chord_gen_path
    
    new_score.dump_midi(output_path)
    
    print(f"Converted chord_gen TPQ from {chord_tpq} to {original_tpq}")
    print(f"Output saved to: {output_path}")
    
    return output_path


def quantize_melody_to_16th(input_path, output_path=None):
    """
    Quantize melody notes to nearest 1/16 note positions, preserving original TPQ.
    
    Args:
        input_path: Path to the input MIDI file
        output_path: Optional output path, defaults to input path with '_quantized' suffix
    
    Returns:
        str: Path to the output file
    """
    from symusic import Note, Track
    
    score = Score(input_path, ttype="tick")
    tpq = score.ticks_per_quarter
    ticks_per_16th = tpq / 4  # 1/16 note = 1/4 quarter note
    
    print(f"=== Quantizing melody to 1/16 notes ===")
    print(f"TPQ: {tpq}")
    print(f"Ticks per 16th note: {ticks_per_16th}")
    
    new_score = Score(ttype="tick")
    new_score.ticks_per_quarter = tpq
    new_score.tempos = score.tempos
    
    total_notes = 0
    for track in score.tracks:
        new_track = Track(name=track.name)
        quantized_notes = []
        for note in track.notes:
            # Quantize start time to nearest 1/16 note
            quantized_start = int(round(note.start / ticks_per_16th) * ticks_per_16th)
            # Quantize duration (minimum 1/16 note)
            quantized_duration = max(
                int(ticks_per_16th),
                int(round(note.duration / ticks_per_16th) * ticks_per_16th)
            )
            quantized_notes.append((quantized_start, quantized_duration, note.pitch, note.velocity))

        # Deduplicate: if two notes snap to the same start tick, keep only the last one
        # (earlier note is a shorter fragment that got snapped forward)
        quantized_notes.sort(key=lambda x: x[0])
        deduped = []
        for start, dur, pitch, vel in quantized_notes:
            if deduped and deduped[-1][0] == start:
                deduped[-1] = [start, dur, pitch, vel]
            else:
                deduped.append([start, dur, pitch, vel])

        # Clip each note's duration so it does not overlap the next note's start
        for i, note in enumerate(deduped):
            if i + 1 < len(deduped):
                note[1] = max(1, min(note[1], deduped[i + 1][0] - note[0]))
            new_track.notes.append(Note(time=note[0], duration=note[1], pitch=note[2], velocity=note[3]))
            total_notes += 1

        new_track.notes.sort(key=lambda n: n.time)
        new_score.tracks.append(new_track)
    
    # Determine output path
    if output_path is None:
        base_path = os.path.splitext(input_path)[0]
        output_path = f"{base_path}_quantized.mid"
    
    new_score.dump_midi(output_path)
    
    print(f"Quantized {total_notes} notes to 1/16 note positions")
    print(f"Output saved to: {output_path}")
    
    return output_path


def requantize_chord_gen_melody(chord_gen_path, output_path=None, grid_resolution=16):
    """
    Re-quantize the melody track in chord_gen output to fix chorderator's rounding errors.
    
    Chorderator introduces small rounding errors (1-2 ticks) when it re-quantizes
    the melody internally. This function fixes those errors by snapping notes
    to the nearest grid position.
    
    Args:
        chord_gen_path: Path to the chord_gen MIDI file
        output_path: Optional output path, defaults to overwriting input
        grid_resolution: Grid resolution in divisions per beat (default 16 = 1/16 beat)
    
    Returns:
        str: Path to the output file
    """
    from symusic import Note, Track
    
    score = Score(chord_gen_path, ttype="tick")
    tpq = score.ticks_per_quarter
    ticks_per_grid = tpq / grid_resolution
    
    new_score = Score(ttype="tick")
    new_score.ticks_per_quarter = tpq
    new_score.tempos = score.tempos
    new_score.time_signatures = score.time_signatures.copy() if score.time_signatures else []
    
    melody_notes_fixed = 0
    
    for i, track in enumerate(score.tracks):
        new_track = Track(name=track.name, program=track.program, is_drum=track.is_drum)
        
        for note in track.notes:
            # Only quantize the melody track (first track, typically)
            # Chord tracks usually have notes at exact positions already
            if i == 0:  # Melody track
                # Snap to nearest grid position
                quantized_start = int(round(note.time / ticks_per_grid) * ticks_per_grid)
                quantized_end = int(round((note.time + note.duration) / ticks_per_grid) * ticks_per_grid)
                quantized_duration = quantized_end - quantized_start
                
                # Ensure minimum duration
                if quantized_duration < ticks_per_grid:
                    quantized_duration = int(ticks_per_grid)
                
                if note.time != quantized_start:
                    melody_notes_fixed += 1
                
                new_note = Note(
                    time=quantized_start,
                    duration=quantized_duration,
                    pitch=note.pitch,
                    velocity=note.velocity
                )
            else:
                # Keep chord track notes as-is
                new_note = Note(
                    time=note.time,
                    duration=note.duration,
                    pitch=note.pitch,
                    velocity=note.velocity
                )
            
            new_track.notes.append(new_note)
        
        new_track.notes.sort(key=lambda n: n.time)
        new_score.tracks.append(new_track)
    
    if output_path is None:
        output_path = chord_gen_path
    
    new_score.dump_midi(output_path)
    
    if melody_notes_fixed > 0:
        print(f"Re-quantized {melody_notes_fixed} melody notes to 1/{grid_resolution} beat grid")
    
    return output_path
