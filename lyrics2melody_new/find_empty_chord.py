"""
Find lyrics_{id} folders whose chord/ directory contains no files.
Checks recursively (no files anywhere inside chord/, including subfolders).
"""
import os
import re

BASE = os.path.dirname(os.path.abspath(__file__))

empty = []
missing = []

entries = sorted(os.listdir(BASE), key=lambda x: int(m.group(1)) if (m := re.match(r"lyrics_(\d+)$", x)) else -1)

for name in entries:
    if not re.match(r"lyrics_\d+$", name):
        continue
    chord_dir = os.path.join(BASE, name, "chord")
    if not os.path.isdir(chord_dir):
        missing.append(name)
        continue
    # Count all files recursively inside chord/
    file_count = sum(len(files) for _, _, files in os.walk(chord_dir))
    if file_count == 0:
        empty.append(name)

print(f"Total lyrics folders checked: {len(empty) + len(missing) + (len(entries) - len(missing) - len(empty))}")
print(f"Missing chord/ folder: {len(missing)}")
print(f"Empty chord/ folder (no files): {len(empty)}")

if empty:
    print("\nFolders with empty chord/:")
    for name in empty:
        print(f"  {name}")

if missing:
    print("\nFolders missing chord/ entirely:")
    for name in missing:
        print(f"  {name}")
