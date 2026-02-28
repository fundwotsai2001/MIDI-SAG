import os

# Create chord mapping dictionary based on actual btc format
chord_map = {
    # Major chords (just root note)
    'C': 'C',
    'C#': 'C#',
    'Db': 'Db', 
    'D': 'D',
    'D#': 'D#',
    'Eb': 'Eb',
    'E': 'E',
    'F': 'F',
    'F#': 'F#',
    'Gb': 'Gb',
    'G': 'G',
    'G#': 'G#',
    'Ab': 'Ab',
    'A': 'A',
    'A#': 'A#',
    'Bb': 'Bb',
    'B': 'B',
    
    # Minor chords (root + :min)
    'C:min': 'C:min',
    'C#:min': 'C#:min',
    'Db:min': 'Db:min',
    'D:min': 'D:min',
    'D#:min': 'D#:min',
    'Eb:min': 'Eb:min',
    'E:min': 'E:min',
    'F:min': 'F:min',
    'F#:min': 'F#:min',
    'Gb:min': 'Gb:min',
    'G:min': 'G:min',
    'G#:min': 'G#:min',
    'Ab:min': 'Ab:min',
    'A:min': 'A:min',
    'A#:min': 'A#:min',
    'Bb:min': 'Bb:min',
    'B:min': 'B:min',
    
    # Seventh chords
    'C:7': 'C:7',
    'C#:7': 'C#:7',
    'Db:7': 'Db:7',
    'D:7': 'D:7',
    'D#:7': 'D#:7',
    'Eb:7': 'Eb:7',
    'E:7': 'E:7',
    'F:7': 'F:7',
    'F#:7': 'F#:7',
    'Gb:7': 'Gb:7',
    'G:7': 'G:7',
    'G#:7': 'G#:7',
    'Ab:7': 'Ab:7',
    'A:7': 'A:7',
    'A#:7': 'A#:7',
    'Bb:7': 'Bb:7',
    'B:7': 'B:7',
    
    # Minor seventh chords
    'C:min7': 'C:min7',
    'C#:min7': 'C#:min7',
    'Db:min7': 'Db:min7',
    'D:min7': 'D:min7',
    'D#:min7': 'D#:min7',
    'Eb:min7': 'Eb:min7',
    'E:min7': 'E:min7',
    'F:min7': 'F:min7',
    'F#:min7': 'F#:min7',
    'Gb:min7': 'Gb:min7',
    'G:min7': 'G:min7',
    'G#:min7': 'G#:min7',
    'Ab:min7': 'Ab:min7',
    'A:min7': 'A:min7',
    'A#:min7': 'A#:min7',
    'Bb:min7': 'Bb:min7',
    'B:min7': 'B:min7',
    
    # Major seventh chords
    'C:maj7': 'C:maj7',
    'C#:maj7': 'C#:maj7',
    'Db:maj7': 'Db:maj7',
    'D:maj7': 'D:maj7',
    'D#:maj7': 'D#:maj7',
    'Eb:maj7': 'Eb:maj7',
    'E:maj7': 'E:maj7',
    'F:maj7': 'F:maj7',
    'F#:maj7': 'F#:maj7',
    'Gb:maj7': 'Gb:maj7',
    'G:maj7': 'G:maj7',
    'G#:maj7': 'G#:maj7',
    'Ab:maj7': 'Ab:maj7',
    'A:maj7': 'A:maj7',
    'A#:maj7': 'A#:maj7',
    'Bb:maj7': 'Bb:maj7',
    'B:maj7': 'B:maj7',
    
    # Diminished chords
    'C:dim': 'C:dim',
    'C#:dim': 'C#:dim',
    'Db:dim': 'Db:dim',
    'D:dim': 'D:dim',
    'D#:dim': 'D#:dim',
    'Eb:dim': 'Eb:dim',
    'E:dim': 'E:dim',
    'F:dim': 'F:dim',
    'F#:dim': 'F#:dim',
    'Gb:dim': 'Gb:dim',
    'G:dim': 'G:dim',
    'G#:dim': 'G#:dim',
    'Ab:dim': 'Ab:dim',
    'A:dim': 'A:dim',
    'A#:dim': 'A#:dim',
    'Bb:dim': 'Bb:dim',
    'B:dim': 'B:dim',
    
    # Augmented chords
    'C:aug': 'C:aug',
    'C#:aug': 'C#:aug',
    'Db:aug': 'Db:aug',
    'D:aug': 'D:aug',
    'D#:aug': 'D#:aug',
    'Eb:aug': 'Eb:aug',
    'E:aug': 'E:aug',
    'F:aug': 'F:aug',
    'F#:aug': 'F#:aug',
    'Gb:aug': 'Gb:aug',
    'G:aug': 'G:aug',
    'G#:aug': 'G#:aug',
    'Ab:aug': 'Ab:aug',
    'A:aug': 'A:aug',
    'A#:aug': 'A#:aug',
    'Bb:aug': 'Bb:aug',
    'B:aug': 'B:aug',
    
    # Sus chords
    'C:sus2': 'C:sus2',
    'C#:sus2': 'C#:sus2',
    'Db:sus2': 'Db:sus2',
    'D:sus2': 'D:sus2',
    'D#:sus2': 'D#:sus2',
    'Eb:sus2': 'Eb:sus2',
    'E:sus2': 'E:sus2',
    'F:sus2': 'F:sus2',
    'F#:sus2': 'F#:sus2',
    'Gb:sus2': 'Gb:sus2',
    'G:sus2': 'G:sus2',
    'G#:sus2': 'G#:sus2',
    'Ab:sus2': 'Ab:sus2',
    'A:sus2': 'A:sus2',
    'A#:sus2': 'A#:sus2',
    'Bb:sus2': 'Bb:sus2',
    'B:sus2': 'B:sus2',
    
    'C:sus4': 'C:sus4',
    'C#:sus4': 'C#:sus4',
    'Db:sus4': 'Db:sus4',
    'D:sus4': 'D:sus4',
    'D#:sus4': 'D#:sus4',
    'Eb:sus4': 'Eb:sus4',
    'E:sus4': 'E:sus4',
    'F:sus4': 'F:sus4',
    'F#:sus4': 'F#:sus4',
    'Gb:sus4': 'Gb:sus4',
    'G:sus4': 'G:sus4',
    'G#:sus4': 'G#:sus4',
    'Ab:sus4': 'Ab:sus4',
    'A:sus4': 'A:sus4',
    'A#:sus4': 'A#:sus4',
    'Bb:sus4': 'Bb:sus4',
    'B:sus4': 'B:sus4',
    
    # No chord
    'N': 'N',
    'None': 'N'
}

quality_mapping = {
    'o': 'dim',
    '+': 'aug',
    'o7': 'dim7',
    '/o7': 'hdim7',
    'o/7': 'hdim7'
}

def convert_chorder_to_btc(chorder_chord):
    """
    Convert chorder chord format to btc format using the chord_map dictionary
    """
    # Handle None or empty chords
    if chorder_chord is None or chorder_chord == 'None':
        return 'N'
    
    # Handle colon format first (C:o/D, C:/o7)
    if ':' in chorder_chord:
        root, quality = chorder_chord.split(':', 1)
        
        # Check if quality contains slash (bass note)
        if '/' in quality and not quality.startswith('/'):
            # This is a bass note case like C:o/D
            chord_quality, bass = quality.split('/', 1)
            # Convert chord quality to BTC format
            quality_btc = quality_mapping.get(chord_quality, chord_quality)
            return f"{root}:{quality_btc}/{bass}"
        else:
            # This is a quality case like C:/o7 or C:o
            # Convert quality to BTC format
            quality_btc = quality_mapping.get(quality, quality)
            return f"{root}:{quality_btc}"
    
    # Handle slash chords (bass notes) - check if bass is a valid note
    if '/' in chorder_chord:
        root_quality, bass = chorder_chord.split('/', 1)
        # Check if bass is a valid note name
        valid_notes = ['C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'F', 'F#', 'Gb', 'G', 'G#', 'Ab', 'A', 'A#', 'Bb', 'B']
        if bass in valid_notes:
            # Convert the root_quality part first
            root_quality_btc = convert_chorder_to_btc(root_quality)
            if root_quality_btc != 'N':
                return f"{root_quality_btc}/{bass}"
        # If bass is not a valid note, treat the whole thing as a chord quality
        # This handles cases like C/o7 where o7 is a quality, not a bass note
        # Extract root from root_quality (e.g., "C" from "C/o7")
        if '/' in root_quality:
            # This shouldn't happen, but handle it
            root = root_quality.split('/')[0]
        else:
            root = root_quality
        quality_btc = quality_mapping.get(bass, bass)
        return f"{root}:{quality_btc}"
    
    # Direct mapping if exists
    if chorder_chord in chord_map:
        return chord_map[chorder_chord]
    
    # If not found in mapping, return 'N'
    print(f"Warning: Unknown chord '{chorder_chord}', returning 'N'")
    return 'N'

def acc2btc(acc_path, btc_path):
    """
    Convert chorder chord file to btc format
    """
    with open(acc_path, "r") as f:
        lines = f.readlines()
    
    btc_lines = []
    for line in lines:
        line = line.strip()
        if line:
            parts = line.split()
            if len(parts) >= 3:
                start_time = parts[0]
                end_time = parts[1]
                chorder_chord = parts[2]
                
                # Convert chorder chord to btc format
                btc_chord = convert_chorder_to_btc(chorder_chord)
                
                btc_lines.append(f"{start_time} {end_time} {btc_chord}")
                # print(f"{start_time} {end_time} {btc_chord}")
    
    # Write to output file
    with open(btc_path, "w") as f:
        f.write('\n'.join(btc_lines) + '\n')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/data/home/fundwotsai/MIDI-SAG/output/chord_txt")
    parser.add_argument("--output_dir", default="/data/home/fundwotsai/MIDI-SAG/output/btc_txt")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(data_dir):
        print(filename)
        if filename.endswith(".txt"):
            acc2btc(os.path.join(data_dir, filename), os.path.join(output_dir, filename))
