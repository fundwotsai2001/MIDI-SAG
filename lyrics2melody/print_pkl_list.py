import pickle
import os 
from pathlib import Path
# Read the pickle file
for root, dirs, files in os.walk("/home/fundwotsai/lyrics2melody/data/StatisticalAttributes/AA_seq_d64"):
    
    for name in files:
        if not name.endswith((".pkl", ".pickle")):
            continue
        path = Path(root) / name
        # print(f"\nLoading: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Print or explore the content
        print(len(data))
