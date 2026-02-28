import chorderator as cdt
import os
import time
import statistics
import tempfile
from miditok import REMI, TokenizerConfig
from symusic import Score
from pretty_midi import PrettyMIDI
from demo_utils import (get_detailed_key_analysis, 
                       get_key_for_cdt, 
                       get_mode_for_cdt, 
                       get_auto_config,
                       preprocess_melody)

# Config
MIDI_DIR = "/home/feiyueh/AccoMontage2/MIDI demos/inputs/midi"
RUNS = 1000
OUTPUT_DIR = tempfile.mkdtemp(prefix="cdt_bench_", dir="./tmp") if os.path.isdir("./tmp") else tempfile.mkdtemp(prefix="cdt_bench_")

# Initialize tokenizer (same as demo.py)
config = TokenizerConfig(num_velocities=16, use_chords=False, use_programs=False)
tokenizer = REMI(config)

def list_midi_files(midi_dir: str):
    files = [os.path.join(midi_dir, f) for f in os.listdir(midi_dir)
             if f.lower().endswith((".mid", ".midi"))]
    files.sort()
    return files

def setup_for_file(midi_path: str, processed_melody_path: str):
    """Setup chorderator with the same logic as demo.py, but exclude timing"""
    # Process the MIDI file (same as demo.py lines 74-88)
    cdt.set_melody(processed_melody_path)
    
    midi_obj = Score(processed_melody_path)
    tokens = tokenizer(midi_obj)
    if len(tokens) == 1:
        tokens = tokens[0]
    
    # Get key analysis
    key_analysis = get_detailed_key_analysis(tokens.tokens)
    cdt_key_attr = get_key_for_cdt(tokens.tokens, key_analysis)
    cdt_mode_attr = get_mode_for_cdt(tokens.tokens, key_analysis)
    
    # Auto-configure
    auto_config = get_auto_config(tokens.tokens)
    
    tempo = PrettyMIDI(processed_melody_path).get_tempo_changes()[1][0]
    
    # Set parameters (same as demo.py lines 93-98)
    cdt_key_value = getattr(cdt.Key, cdt_key_attr)
    cdt_mode_value = getattr(cdt.Mode, cdt_mode_attr)
    cdt.set_meta(tonic=cdt_key_value, mode=cdt_mode_value, tempo=tempo)
    cdt.set_note_shift(auto_config['note_shift'])
    cdt.set_segmentation(auto_config['segmentation'])
    cdt.set_output_style(cdt.Style.POP_STANDARD)
    
    return auto_config, key_analysis

def run_inference(idx: int):
    """Time only the generation call"""
    out_name = f"bench_{idx:04d}_chord_gen.mid"
    # Time only the generation call
    t0 = time.perf_counter()
    chord_gen = cdt.generate_save(output_dir=OUTPUT_DIR, 
                                  chord_output_name=out_name, 
                                  task='chord',
                                  log=False)
    t1 = time.perf_counter()
    return (t1 - t0)

def main():
    midi_files = list_midi_files(MIDI_DIR)
    if not midi_files:
        print("No MIDI files found in MIDI_DIR.")
        return

    print(f"Found {len(midi_files)} MIDI files.")
    print(f"Output tmp dir: {OUTPUT_DIR}")

    # Create temp directory for processed melodies
    processed_melody_dir = tempfile.mkdtemp(prefix="processed_melody_", dir=OUTPUT_DIR)

    # Warm-up (exclude from stats): trigger model load/graph compile once
    first_file = midi_files[0]
    processed_melody_path = os.path.join(processed_melody_dir, "warmup_" + os.path.basename(first_file))
    
    try:
        # Preprocess melody (same as demo.py)
        preprocess_melody(first_file, processed_melody_path)
        setup_for_file(first_file, processed_melody_path)
        _ = run_inference(idx=-1)
        print("Warm-up completed successfully")
    except Exception as e:
        print(f"Warm-up failed: {e}")
        return

    times = []
    successful_runs = 0
    
    # Main loop: cycle through files to reach RUNS
    for i in range(RUNS):
        midi_path = midi_files[i % len(midi_files)]
        midi_name = os.path.basename(midi_path)
        processed_melody_path = os.path.join(processed_melody_dir, f"run_{i:04d}_" + midi_name)
        
        try:
            # Preprocess melody (outside timing)
            preprocess_melody(midi_path, processed_melody_path)
            
            # Setup chorderator (outside timing)
            auto_config, key_analysis = setup_for_file(midi_path, processed_melody_path)
            
            # Time only the inference
            dt = run_inference(idx=i)
            times.append(dt)
            successful_runs += 1
            
            if (successful_runs) % 50 == 0:
                avg_time = statistics.mean(times)
                print(f"[{successful_runs}/{RUNS}] {midi_name}: {dt*1000:.2f} ms, avg={avg_time*1000:.2f} ms")
                print(f"  Key: {key_analysis['key']} {key_analysis['mode']}, Segmentation: {auto_config['segmentation']}")
                
        except Exception as e:
            print(f"[{i+1}/{RUNS}] Failed on {midi_name}: {e}")

    if not times:
        print("No successful runs to report.")
        return

    times_sorted = sorted(times)
    n = len(times_sorted)
    def pct(p):
        k = max(0, min(n - 1, int(round(p * (n - 1)))))
        return times_sorted[k]

    mean = statistics.mean(times_sorted)
    median = statistics.median(times_sorted)
    stdev = statistics.pstdev(times_sorted) if n > 1 else 0.0
    t_min = times_sorted[0]
    t_max = times_sorted[-1]
    p90 = pct(0.90)
    p95 = pct(0.95)
    p99 = pct(0.99)

    print(f"\n=== Chorderator chord generation inference time (seconds) ===")
    print(f"successful runs: {n}/{RUNS}")
    print(f"mean           : {mean:.6f} s ({mean*1000:.2f} ms)")
    print(f"median         : {median:.6f} s ({median*1000:.2f} ms)")
    print(f"stdev          : {stdev:.6f} s ({stdev*1000:.2f} ms)")
    print(f"min/max        : {t_min:.6f} s / {t_max:.6f} s ({t_min*1000:.2f} / {t_max*1000:.2f} ms)")
    print(f"p90 / p95 / p99: {p90:.6f} s / {p95:.6f} s / {p99:.6f} s")
    print(f"                 ({p90*1000:.2f} / {p95*1000:.2f} / {p99*1000:.2f} ms)")

    # Clean up temp files
    import shutil
    shutil.rmtree(OUTPUT_DIR)
    print(f"Cleaned up temp directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()