import argparse
import chorderator as cdt
import os
import json
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
                       quantize_melody_to_16th,
                       reassign_global_tempo,
                       load_beat_times,
                       estimate_tempo_from_beats,
                       adjust_midi_to_beats,
                       requantize_chord_gen_melody,
                       fill_none_chords_in_txt
                       )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate chord progression from a single vocal MIDI file.')
    parser.add_argument('--midi_path', type=str, required=True, help='Path to input vocal MIDI file')
    parser.add_argument('--beat_file', type=str, default=None, help='Path to beat times txt file (optional)')
    parser.add_argument('--output_dir', type=str, default='output_SOME', help='Base output directory')
    parser.add_argument('--beat_subdivision', type=int, default=1,
                        help='Beat subdivision: 1=quarter, 2=8th, 4=16th notes (default: 1)')
    parser.add_argument('--chord_style', type=str, default='POP_STANDARD',
                        choices=['POP_STANDARD', 'POP_COMPLEX', 'DARK', 'RANDB', 'NOCONSTRAINT'],
                        help='Output chord style (default: POP_STANDARD)')
    args = parser.parse_args()

    input_melody_path = args.midi_path
    BEAT_SUBDIVISION = args.beat_subdivision

    # Derive song name from filename stem
    song_name = os.path.splitext(os.path.basename(input_melody_path))[0]
    demo_name = song_name.replace(" ", "_").replace("/", "_")

    # Create output directory structure
    output_base_dir = args.output_dir
    processed_melody_dir = os.path.join(output_base_dir, "processed_melody")
    chord_gen_dir = os.path.join(output_base_dir, "chord_gen")
    chord_gen_filled_dir = os.path.join(output_base_dir, "chord_gen_filled_empty")
    chord_gen_quantized_dir = os.path.join(output_base_dir, "chord_gen_quantized")
    chord_txt_with_none_dir = os.path.join(output_base_dir, "chord_txt_with_None")
    chord_txt_dir = os.path.join(output_base_dir, "chord_txt")
    btc_txt_dir = os.path.join(output_base_dir, "btc_txt")

    os.makedirs(processed_melody_dir, exist_ok=True)
    os.makedirs(chord_gen_dir, exist_ok=True)
    os.makedirs(chord_gen_filled_dir, exist_ok=True)
    os.makedirs(chord_gen_quantized_dir, exist_ok=True)
    os.makedirs(chord_txt_with_none_dir, exist_ok=True)
    os.makedirs(chord_txt_dir, exist_ok=True)
    os.makedirs(btc_txt_dir, exist_ok=True)

    print(f"\n=== Processing: {song_name} ===")
    print(f"Output will be saved to: {output_base_dir}/")

    processed_melody_path = os.path.join(processed_melody_dir, f"{demo_name}.mid")

    beat_times = None
    estimated_bpm = None
    beat_file_path = args.beat_file
    print("beat_file_path", beat_file_path)
    print("os.path.exists(beat_file_path)", os.path.exists(beat_file_path) if beat_file_path else False)
    try:
        if beat_file_path and os.path.exists(beat_file_path):
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
        cdt.set_output_style(getattr(cdt.Style, args.chord_style))

        # Generate chord progression
        chord_gen_output = os.path.join(chord_gen_dir, f"{demo_name}_chord_gen.mid")
        chord_gen = cdt.generate_save(output_dir=chord_gen_dir,
                                      chord_output_name=f"{demo_name}_chord_gen.mid",
                                      task='chord',
                                      log=False)

        # Align chord_gen TPQ with original melody TPQ
        align_chord_gen_tpq(processed_melody_path, chord_gen_output)

        # Re-quantize melody track to fix chorderator's rounding errors
        requantize_chord_gen_melody(chord_gen_output)

        # Fill empty bars
        empty_bars = auto_config['analysis']['empty_bars']
        filled_output = os.path.join(chord_gen_filled_dir, f"{demo_name}_chord_gen_filled_empty_bars.mid")
        fill_empty_bars_with_chords(
            processed_melody_path,
            chord_gen_output,
            empty_bars,
            filled_output
        )

        # Create quantized (1/16 note) version
        quantized_output = os.path.join(chord_gen_quantized_dir, f"{demo_name}_chord_gen_quantized.mid")
        quantize_melody_to_16th(filled_output, quantized_output)

        # Export chord text with None values
        txt_with_none = os.path.join(chord_txt_with_none_dir, f"{demo_name}_chord_gen.txt")
        export_chords_txt_chorder(filled_output, txt_with_none, beat_times=beat_times,
                                  beat_subdivision=BEAT_SUBDIVISION)

        # Fill None chords
        txt_file = os.path.join(chord_txt_dir, f"{demo_name}_chord_gen.txt")
        fill_none_chords_in_txt(txt_with_none, txt_file)

        # Convert to BTC format
        btc_file = os.path.join(btc_txt_dir, f"{demo_name}_chord_gen.txt")
        acc2btc(txt_file, btc_file)

        print(f"✓ Successfully processed: {song_name}")
        print(f"  Key: {key_analysis['key']} {key_analysis['mode']} (confidence: {key_analysis['confidence']:.2f})")
        print(f"  Tempo: {tempo:.1f} BPM (from beats: {beat_times is not None})")
        print(f"  Chord Text: {txt_file}")
        print(f"  BTC Text:   {btc_file}")

    except Exception as e:
        print(f"✗ Failed to process: {song_name}")
        print(f"  Error: {str(e)}")
        raise
