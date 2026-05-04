import pickle
import os 
from pathlib import Path
# Read the pickle file
lyrics_lengths = []
for root, dirs, files in os.walk("/home/fundwotsai/lyrics2melody/share_StatisticalAttributes/AA_seq_d64"):
    for name in files:
        if not name.endswith((".pkl", ".pickle")):
            continue
        path = Path(root) / name
        # print(f"\nLoading: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Print or explore the content
        lyrics_lengths.append(len(data))
unique = set(lyrics_lengths)
print(unique)