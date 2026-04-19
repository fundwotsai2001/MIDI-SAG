#!/usr/bin/env python3
"""Dump the full contents of a pickle file.

Usage:
  python dump_pkl.py path/to/file.pkl                # print to stdout
  python dump_pkl.py path/to/file.pkl -o out.txt     # write to file
"""

from __future__ import annotations

import argparse
import pickle
import pprint
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pkl", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Write the dump to this file instead of stdout.")
    ap.add_argument("--width", type=int, default=120,
                    help="pprint line width (default 120).")
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)

    stream = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        pprint.pprint(data, stream=stream, width=args.width)
    finally:
        if args.output:
            stream.close()
            print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
