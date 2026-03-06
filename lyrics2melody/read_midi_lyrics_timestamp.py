from mido import MidiFile, merge_tracks
import os
import sys
mid_folder = sys.argv[1]
def fmt_time(sec: float) -> str:
    m = int(sec // 60)
    s = sec - 60 * m
    return f'{m:02d}:{s:06.3f}'  # mm:ss.mmm
mid_paths = [
    f for f in os.listdir(mid_folder)
    if f.lower().endswith(".mid")
]
for mid_path in mid_paths:
    print("mid_path", mid_path)
    mid_path = os.path.join(mid_folder, mid_path)
    mid = MidiFile(mid_path)
    out_path = mid_path.replace('.mid', '_times.txt')
    # Merge all tracks so tempo and lyrics are seen in a single time-ordered stream.
    merged = merge_tracks(mid.tracks)
    tpb = mid.ticks_per_beat
    tempo = 500_000  # default 120 BPM
    t_sec = 0.0

    sentence_chunks = []
    sentence_start  = None
    lines_written   = 0

    with open(out_path, 'w', encoding='utf-8') as f:
        for msg in merged:
            # Advance "clock"
            if msg.time:
                t_sec += (msg.time * tempo) / (tpb * 1_000_000.0)

            # Tempo changes
            if msg.is_meta and msg.type == 'set_tempo':
                tempo = msg.tempo
                continue

            # Lyrics events
            if msg.is_meta and msg.type in ('lyrics', 'lyric'):
                raw = msg.text
                b = raw.encode('latin-1', errors='ignore')
                proper = b.decode('utf-8', errors='ignore')
                cleaned = proper.replace('*', '#')

                ends_sentence = cleaned.endswith('.')
                token = cleaned[:-1] if ends_sentence else cleaned

                if token:
                    if sentence_start is None:
                        sentence_start = t_sec
                    sentence_chunks.append(token)

                if ends_sentence:
                    sentence = ''.join(sentence_chunks).strip()
                    if sentence:
                        # here: use t_sec as the end of this sentence
                        f.write(f'{fmt_time(sentence_start)}\t{fmt_time(t_sec)}\t{sentence}\n')
                        lines_written += 1
                    sentence_chunks.clear()
                    sentence_start = None

        # Flush trailing sentence if any
        if sentence_chunks:
            sentence = ''.join(sentence_chunks).strip()
            if sentence:
                # t_sec at this point is already the "end of last note/event"
                f.write(f'{fmt_time(sentence_start or 0.0)}\t{fmt_time(t_sec)}\t{sentence}\n')
                lines_written += 1


    print(f'Wrote {lines_written} timed sentences to {out_path!r}')
