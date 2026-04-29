from mido import MidiFile
import sys

mid_path  = sys.argv[1]
out_path  = sys.argv[2]

mid = MidiFile(mid_path)
with open(out_path, 'w', encoding='utf-8') as f:
    for i, track in enumerate(mid.tracks):
        for msg in track:
            if msg.is_meta and msg.type in ('lyrics', 'lyric'):
                # recover the raw bytes → proper UTF-8 string
                raw    = msg.text
                b      = raw.encode('latin-1', errors='ignore')
                proper = b.decode('utf-8', errors='ignore')

                # now apply your replacements:
                #   remove all dots, replace '*' with '#'
                cleaned = proper.replace('*', '#')
                if cleaned.endswith('.'):
                    # drop the trailing dot, then newline
                    f.write(cleaned[:-1] + '\n')
                else:
                    f.write(cleaned)
                

print(f'Wrote cleaned lyrics to {out_path!r}')
