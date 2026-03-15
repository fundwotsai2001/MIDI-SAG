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
                       requantize_chord_gen_melody
                       )

if __name__ == '__main__':
    # Process all MIDI files in the directory
    midi_dir = "/home/feiyueh/AccoMontage2/MIDI demos/inputs/rofo"
    
    # Create output directory structure
    output_base_dir = "batch_processing_results_rofo"
    processed_melody_dir = os.path.join(output_base_dir, "processed_melody")
    chord_gen_dir = os.path.join(output_base_dir, "chord_gen")
    chord_gen_filled_dir = os.path.join(output_base_dir, "chord_gen_filled_empty")
    chord_gen_quantized_dir = os.path.join(output_base_dir, "chord_gen_quantized")
    chord_txt_dir = os.path.join(output_base_dir, "chord_txt")
    
    # Create directories if they don't exist
    os.makedirs(processed_melody_dir, exist_ok=True)
    os.makedirs(chord_gen_dir, exist_ok=True)
    os.makedirs(chord_gen_filled_dir, exist_ok=True)
    os.makedirs(chord_gen_quantized_dir, exist_ok=True)
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
    
    # Get all MIDI files
    midi_files = [f for f in os.listdir(midi_dir) if f.endswith('.mid') or f.endswith('.midi')]
    results['summary']['total_files'] = len(midi_files)
    
    print(f"Found {len(midi_files)} MIDI files to process")
    print(f"Output will be saved to: {output_base_dir}/")
    print(f"  - chord_gen/: Original chord generation results")
    print(f"  - chord_gen_filled_empty/: Filled empty bars results")
    print(f"  - chord_gen_quantized/: Quantized (1/16 note) version")
    print(f"  - chord_txt/: Chord text files")
    
    for midi_file in midi_files:
        try:
            print(f"\n=== Processing: {midi_file} ===")
            
            input_melody_path = os.path.join(midi_dir, midi_file)
            processed_melody_path = os.path.join(processed_melody_dir, midi_file)
            # preprocess the melody, leave only one track
            preprocess_melody(input_melody_path, processed_melody_path)
            
            demo_name = midi_file.split('.')[0]
            
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
            
            # Auto-configure
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
            
            # Export chord text - save to chord_txt directory
            txt_file = os.path.join(chord_txt_dir, f"{demo_name}_chord_gen_filled_empty_bars.txt")
            export_chords_txt_chorder(filled_output, txt_file)
            
            # Store successful result
            file_result = {
                'filename': midi_file,
                'demo_name': demo_name,
                'key': key_analysis['key'],
                'mode': key_analysis['mode'],
                'confidence': key_analysis['confidence'],
                'total_bars': auto_config['analysis']['total_bars'],
                'empty_bars': auto_config['analysis']['empty_bars'],
                'content_bars': auto_config['analysis']['content_bars'],
                'note_shift': auto_config['note_shift'],
                'segmentation': auto_config['segmentation'],
                'chord_gen_midi': chord_gen_output,
                'chord_gen_filled_midi': filled_output,
                'chord_gen_quantized_midi': quantized_output,
                'chord_txt': txt_file,
                'status': 'success'
            }
            results['processed_files'].append(file_result)
            results['summary']['successful'] += 1
            
            print(f"✓ Successfully processed: {midi_file}")
            print(f"  Key: {key_analysis['key']} {key_analysis['mode']} (confidence: {key_analysis['confidence']:.2f})")
            print(f"  Chord Gen: {chord_gen_output}")
            print(f"  Filled Empty: {filled_output}")
            print(f"  Quantized: {quantized_output}")
            print(f"  Chord Text: {txt_file}")
            
        except Exception as e:
            # Store failed result
            error_result = {
                'filename': midi_file,
                'error': str(e),
                'status': 'failed'
            }
            results['failed_files'].append(error_result)
            results['summary']['failed'] += 1
            
            print(f"✗ Failed to process: {midi_file}")
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
