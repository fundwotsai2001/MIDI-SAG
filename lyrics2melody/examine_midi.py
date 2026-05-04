from mido import MidiFile, merge_tracks, tick2second

mid = MidiFile("/volume/fundwo-test/lyrics2melody/share/不像情歌-小贱-100-B.mid")
print(mid.type, len(mid.tracks), mid.ticks_per_beat)

tempo = 500_000  # default 120 BPM
t = 0.0
for msg in merge_tracks(mid.tracks):
    t += tick2second(msg.time, mid.ticks_per_beat, tempo)
    if msg.type == "set_tempo":
        tempo = msg.tempo
    print(f"{t:8.3f}s  {msg}")
