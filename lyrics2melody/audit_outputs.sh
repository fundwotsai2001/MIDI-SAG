#!/usr/bin/env bash
set -euo pipefail

# === Config (edit if your paths or filenames differ) =========================
ROOT=/volume/fundwo-test/lyrics2melody
OUT_BASE="$ROOT/generated_midi"                  # where <STEM>/ lives
SUMMARY="$ROOT/audit_summary.csv"               # CSV report output

# Required files per song (relative to <STEM>/)
REQ_FILES=(
  "sample01.mid"
  "sample01.json"
  "sample01.txt"
  "sample01_times.txt"
  "sample01_struct_time.json"
  "sample01_struct_label.json"
  "sample01.wav"
)

# Optional dirs that should contain at least one file (relative to <STEM>/)
REQ_NONEMPTY_DIRS=(
  "chord/chord_txt"
  "chord/chord_btc_txt"
)
# ============================================================================

fail=0
echo "song_stem,file_or_dir,type,ok,size_bytes,mtime" > "$SUMMARY"

shopt -s nullglob
stems=( "$OUT_BASE"/* )
if (( ${#stems[@]} == 0 )); then
  echo "[audit] No song folders found under $OUT_BASE"; exit 2
fi

for stemdir in "${stems[@]}"; do
  [[ -d "$stemdir" ]] || continue
  stem="$(basename "$stemdir")"

  # 1) check required files
  for rel in "${REQ_FILES[@]}"; do
    p="$stemdir/$rel"
    if [[ -s "$p" ]]; then
      size=$(stat -c '%s' "$p" 2>/dev/null || stat -f '%z' "$p")
      mtime=$(date -d @"$(stat -c '%Y' "$p" 2>/dev/null || stat -f '%m' "$p")" '+%Y-%m-%d %H:%M:%S')
      echo "$stem,$rel,file,OK,$size,$mtime" >> "$SUMMARY"
    else
      echo "$stem,$rel,file,MISSING,0," >> "$SUMMARY"
      echo "❌  [$stem] missing file: $rel"
      fail=1
    fi
  done

  # 2) check non-empty directories
  for rel in "${REQ_NONEMPTY_DIRS[@]}"; do
    d="$stemdir/$rel"
    if [[ -d "$d" ]]; then
      # count non-hidden files
      shopt -s nullglob dotglob
      files=( "$d"/* )
      shopt -u dotglob
      if (( ${#files[@]} > 0 )); then
        echo "$stem,$rel,dir,OK,${#files[@]} files," >> "$SUMMARY"
      else
        echo "$stem,$rel,dir,EMPTY,0," >> "$SUMMARY"
        echo "❌  [$stem] empty dir: $rel"
        fail=1
      fi
    else
      echo "$stem,$rel,dir,MISSING,0," >> "$SUMMARY"
      echo "❌  [$stem] missing dir: $rel"
      fail=1
    fi
  done
done

echo
echo "Audit summary written to: $SUMMARY"
if (( fail == 0 )); then
  echo "✅ All required artifacts present."
else
  echo "⚠️  Some artifacts are missing (see lines with MISSING/EMPTY above and in $SUMMARY)."
fi
exit $fail
