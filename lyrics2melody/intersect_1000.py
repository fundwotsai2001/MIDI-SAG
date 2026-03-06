import os

A = "/home/fundwotsai/lyrics2melody/sentence_struct"
B = "/home/fundwotsai/lyrics2melody/share_StatisticalAttributes/AA_seq_d64"

# list files
files_A = set(os.listdir(A))
files_B = set(os.listdir(B))

# intersection
common_files = files_A & files_B

print("Files in both A and B:")
for f in sorted(common_files):
    print(f)
import shutil

output = "/home/fundwotsai/lyrics2melody/sentence_struct_1000"
os.makedirs(output, exist_ok=True)

for f in common_files:
    src = os.path.join(A, f)
    dst = os.path.join(output, f)
    shutil.copy2(src, dst)   # or shutil.move(src, dst)
