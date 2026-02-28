import chorderator as cdt
import os
import json
from datetime import datetime
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
                       quantize_melody_to_16th,
                       reassign_global_tempo,
                       load_beat_times,
                       estimate_tempo_from_beats,
                       adjust_midi_to_beats,
                       requantize_chord_gen_melody,
                       fill_none_chords_in_txt
                       )

if __name__ == '__main__':
    # ============================================
    # CONFIGURATION
    # ============================================
    
    # Process all MIDI files in the directory
    # Each song is in a subdirectory with a vocals.mid file
    midi_base_dir = "/home/b06611012/fundwotsai/MUSDB18_wav/conditions_SOME_predicted_vocal_midi"
    
    # Beat times directory (from SingNet or other beat detection)
    # Each song subdirectory should contain a beat_times.txt file
    beat_base_dir = "/home/b06611012/fundwotsai/MUSDB18_wav/conditions_SingNet_prediced_vocal_beat_updated_v11"
    
    # Use beat times for tempo estimation (recommended)
    USE_BEAT_TIMES = True
    
    # Beat subdivision: what note value does each beat_time represent?
    # 1 = quarter notes (default), 2 = 8th notes, 4 = 16th notes
    # Set to 2 if your beat_times are 8th notes (e.g., conditions_SingNet_prediced_vocal_beat_updated)
    BEAT_SUBDIVISION = 1  # Updated beat_times seem to be 8th notes
    
    # ============================================
    
    # Create output directory structure
    output_base_dir = "batch_processing_results_MUSDB18_updated_v11"
    processed_melody_dir = os.path.join(output_base_dir, "processed_melody")
    chord_gen_dir = os.path.join(output_base_dir, "chord_gen")
    chord_gen_filled_dir = os.path.join(output_base_dir, "chord_gen_filled_empty")
    chord_gen_quantized_dir = os.path.join(output_base_dir, "chord_gen_quantized")
    chord_txt_with_none_dir = os.path.join(output_base_dir, "chord_txt_with_None")
    chord_txt_dir = os.path.join(output_base_dir, "chord_txt")
    
    # Create directories if they don't exist
    os.makedirs(processed_melody_dir, exist_ok=True)
    os.makedirs(chord_gen_dir, exist_ok=True)
    os.makedirs(chord_gen_filled_dir, exist_ok=True)
    os.makedirs(chord_gen_quantized_dir, exist_ok=True)
    os.makedirs(chord_txt_with_none_dir, exist_ok=True)
    os.makedirs(chord_txt_dir, exist_ok=True)
    
    # Data structure to store results
    results = {
        'processed_files': [],
        'failed_files': [],
        'summary': {
            'total_files': 0,
            'successful': 0,
            'failed': 0
        }
    }
    
    # Get all MIDI files from subdirectories (each song folder contains vocals.mid)
    midi_files = []
    for song_dir in os.listdir(midi_base_dir):
        song_path = os.path.join(midi_base_dir, song_dir)
        if os.path.isdir(song_path):
            vocals_midi = os.path.join(song_path, "vocals.mid")
            if os.path.exists(vocals_midi):
                # Store tuple of (song_name, full_path)
                midi_files.append((song_dir, vocals_midi))
    
    results['summary']['total_files'] = len(midi_files)
    
    print(f"Found {len(midi_files)} MIDI files to process")
    print(f"Output will be saved to: {output_base_dir}/")
    print(f"  - chord_gen/: Original chord generation results")
    print(f"  - chord_gen_filled_empty/: Filled empty bars results")
    print(f"  - chord_gen_quantized/: Quantized (1/16 note) version")
    print(f"  - chord_txt/: Chord text files")
    
    for song_name, input_melody_path in midi_files:
        try:
            print(f"\n=== Processing: {song_name} ===")
            
            # Use song name as the base name for output files
            demo_name = song_name.replace(" ", "_").replace("/", "_")
            processed_melody_path = os.path.join(processed_melody_dir, f"{demo_name}.mid")
            
            # Check for beat times file
            beat_file_path = os.path.join(beat_base_dir, song_name, "vocals_beat_times.txt")
            beat_times = None
            estimated_bpm = None
            print("beat_file_path", beat_file_path)
            print("os.path.exists(beat_file_path)", os.path.exists(beat_file_path))
            if USE_BEAT_TIMES and os.path.exists(beat_file_path):
                print(f"Found beat times: {beat_file_path}")
                beat_times = load_beat_times(beat_file_path)
                tempo_info = estimate_tempo_from_beats(beat_times)
                estimated_bpm = tempo_info['tempo']
                print(f"Tempo from beats: {estimated_bpm:.1f} BPM (bar duration: {tempo_info['bar_duration']:.3f}s)")
                
                # First, do basic preprocessing (leave one track)
                score = Score(input_melody_path, ttype="tick")
                for i, track in enumerate(score.tracks):
                    if track.notes:
                        score.tracks = [track]
                        break
                score.dump_midi(processed_melody_path)
                
                # Adjust tempo while preserving absolute note timing (seconds)
                adjust_midi_to_beats(processed_melody_path, processed_melody_path, beat_times, 
                                    beat_subdivision=BEAT_SUBDIVISION)
            else:
                print(f"No beat times found, using standard preprocessing")
                # Preprocess the melody (leave only one track, estimate/set tempo)
                estimated_bpm = preprocess_melody(input_melody_path, processed_melody_path, target_bpm=None)
            
            # Process the MIDI file
            cdt.set_melody(processed_melody_path)
            print(f"processed_melody_path: {processed_melody_path}")
            midi_obj = Score(processed_melody_path)
            tokens = tokenizer(midi_obj)
            if len(tokens) == 1:
                tokens = tokens[0]
            
            # Get key analysis
            key_analysis = get_detailed_key_analysis(tokens.tokens)
            cdt_key_attr = get_key_for_cdt(tokens.tokens, key_analysis)
            cdt_mode_attr = get_mode_for_cdt(tokens.tokens, key_analysis)
            
            # Auto-configure (use MIDI file for accurate bar counting)
            auto_config = get_auto_config(tokens.tokens, midi_path=processed_melody_path)
            print(f"auto_config: {auto_config}")

            tempo = PrettyMIDI(processed_melody_path).get_tempo_changes()[1][0]
            print(f"tempos: {tempo}")
            # Set parameters
            cdt_key_value = getattr(cdt.Key, cdt_key_attr)
            cdt_mode_value = getattr(cdt.Mode, cdt_mode_attr)
            cdt.set_meta(tonic=cdt_key_value, mode=cdt_mode_value, tempo=tempo)
            cdt.set_note_shift(auto_config['note_shift'])
            cdt.set_segmentation(auto_config['segmentation'])
            cdt.set_output_style(cdt.Style.POP_STANDARD)
            
            # Generate chord progression - save to chord_gen directory
            chord_gen_output = os.path.join(chord_gen_dir, f"{demo_name}_chord_gen.mid")
            chord_gen = cdt.generate_save(output_dir=chord_gen_dir, 
                                            chord_output_name=f"{demo_name}_chord_gen.mid", 
                                            task='chord',
                                            log=False)
            
            # Align chord_gen TPQ with original melody TPQ
            align_chord_gen_tpq(processed_melody_path, chord_gen_output)
            
            # Re-quantize melody track to fix chorderator's rounding errors
            requantize_chord_gen_melody(chord_gen_output)
            
            # Fill empty bars and sync tempo - save to chord_gen_filled_empty directory
            empty_bars = auto_config['analysis']['empty_bars']
            filled_output = os.path.join(chord_gen_filled_dir, f"{demo_name}_chord_gen_filled_empty_bars.mid")
            fill_empty_bars_with_chords(
                processed_melody_path,
                chord_gen_output, 
                empty_bars,
                filled_output
            )
            # sync_output_tempo_with_input(processed_melody_path, [filled_output])
            
            # Create quantized (1/16 note) version - save to chord_gen_quantized directory
            quantized_output = os.path.join(chord_gen_quantized_dir, f"{demo_name}_chord_gen_quantized.mid")
            quantize_melody_to_16th(filled_output, quantized_output)
            
            # Export chord text with None values - save to chord_txt_with_None directory
            # Pass beat_times to remap chord times to actual beat positions
            txt_with_none = os.path.join(chord_txt_with_none_dir, f"{demo_name}_chord_gen.txt")
            export_chords_txt_chorder(filled_output, txt_with_none, beat_times=beat_times, 
                                     beat_subdivision=BEAT_SUBDIVISION)
            
            # Fill None chords and save to chord_txt directory
            txt_file = os.path.join(chord_txt_dir, f"{demo_name}_chord_gen.txt")
            fill_none_chords_in_txt(txt_with_none, txt_file)
            
            # Store successful result
            file_result = {
                'filename': song_name,
                'demo_name': demo_name,
                'key': key_analysis['key'],
                'mode': key_analysis['mode'],
                'confidence': key_analysis['confidence'],
                'tempo': float(tempo),
                'estimated_bpm': float(estimated_bpm) if estimated_bpm else None,
                'used_beat_times': beat_times is not None,
                'total_bars': auto_config['analysis']['total_bars'],
                'empty_bars': auto_config['analysis']['empty_bars'],
                'content_bars': auto_config['analysis']['content_bars'],
                'note_shift': auto_config['note_shift'],
                'segmentation': auto_config['segmentation'],
                'chord_gen_midi': chord_gen_output,
                'chord_gen_filled_midi': filled_output,
                'chord_gen_quantized_midi': quantized_output,
                'chord_txt': txt_file,
                'chord_txt_with_none': txt_with_none,
                'status': 'success'
            }
            results['processed_files'].append(file_result)
            results['summary']['successful'] += 1
            
            print(f"✓ Successfully processed: {song_name}")
            print(f"  Key: {key_analysis['key']} {key_analysis['mode']} (confidence: {key_analysis['confidence']:.2f})")
            print(f"  Tempo: {tempo:.1f} BPM (from beats: {beat_times is not None})")
            print(f"  Chord Text: {txt_file}")
            
        except Exception as e:
            # Store failed result
            error_result = {
                'filename': song_name,
                'error': str(e),
                'status': 'failed'
            }
            results['failed_files'].append(error_result)
            results['summary']['failed'] += 1
            
            print(f"✗ Failed to process: {song_name}")
            print(f"  Error: {str(e)}")
    
    # Print summary
    print(f"\n=== Processing Summary ===")
    print(f"Total files: {results['summary']['total_files']}")
    print(f"Successful: {results['summary']['successful']}")
    print(f"Failed: {results['summary']['failed']}")
    
    # Print successful files
    if results['processed_files']:
        print(f"\n=== Successfully Processed Files ===")
        for file_result in results['processed_files']:
            print(f"- {file_result['filename']}: {file_result['key']} {file_result['mode']} ({file_result['confidence']:.2f})")
    
    # Print failed files
    if results['failed_files']:
        print(f"\n=== Failed Files ===")
        for file_result in results['failed_files']:
            print(f"- {file_result['filename']}: {file_result['error']}")
    
    print(f"\nAll processing completed!")
    
    # Save results to JSON file
    results['processing_time'] = datetime.now().isoformat()
    results_file = os.path.join(output_base_dir, 'processing_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {results_file}")
