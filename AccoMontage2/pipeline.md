# Demo Pipeline 說明

## 輸入資料

| 輸入 | 說明 |
|------|------|
| `vocals.mid` | 從 performance audio 取出的 vocal pitch，tempo 寫死為 120 BPM（不正確）。音符的 tick 位置是依據錯誤的 120 BPM 排的 |
| `vocals_beat_times.txt` | beat detection（SingNet）的結果。每行一個 beat 的 onset 秒數。`BEAT_SUBDIVISION=1` 表示每行是四分音符；`=2` 表示每行是八分音符。**必要輸入，缺少則跳過該歌曲並記錄錯誤** |

---

## 當前 Pipeline（`some_demo.py`）

```
vocals.mid (120 BPM 錯誤)          vocals_beat_times.txt
        │                                    │
        └──────────────┬──────────────────────┘
                       ▼
[1] 剝出單一 melody track
    ─ 用 symusic 讀原始 MIDI，保留第一個有音符的 track
    ─ 輸出 processed_melody/ (仍是 120 BPM)
                       │
                       ▼
[2] detect_downbeat_phase()   ← demo_utils.py
    ─ 讀 MIDI 音符 onset（秒）
    ─ 對每個音符找最近的 beat_times index k
    ─ 統計 k % entries_per_bar 的分布
    ─ 出現最多的 position = downbeat（前奏因無 vocal notes，不影響結果）
    ─ 輸出 downbeat_phase（整數，beat_times[downbeat_phase] 是第一個 bar 1 beat 1）
                       │
                       ▼
[3] warp_midi_to_beats()   ← demo_utils.py
    ─ 輸入 beat_times_aligned = beat_times[downbeat_phase:]
      （從第一個 downbeat 開始，tick 0 = bar 1 beat 1）
    ─ 對每個音符：把原始 120 BPM 下的 tick 換算成秒，
      找該秒數落在哪個 beat interval，計算 beat-relative 位置
    ─ 重新映射到新的 tick（在平均 BPM 下的格子）
    ─ quantize_to_grid=True, grid_resolution=16（snap 到 1/16 note 格）
    ─ downbeat 前的 pickup notes → 負 tick → 被 shift 到 bar 0，效果上被丟棄
    ─ 輸出 processed_melody/（已對齊 beat grid，bar 邊界正確）
                       │
                       ▼
[4] REMI tokenization（miditok）
    ─ 現在 bar 邊界已正確對齊 beat grid → REMI 抓到的 bar 是準的
    ─ 輸出 bar tokens、note tokens
                       │
                       ▼
[5] Key analysis + Auto-config
    ─ get_detailed_key_analysis(tokens)  → tonic, mode
    ─ get_auto_config(tokens, midi_path) → segmentation, note_shift, empty_bars
                       │
                       ▼
[6] CDT 和弦生成（chorderator）
    ─ cdt.set_meta(tonic, mode, tempo=estimated_bpm)
      ← tempo 直接用 estimate_tempo_from_beats() 算出的平均 BPM，
        不從 MIDI tempo event 讀（避免 warp 時寫入值的 rounding 誤差）
    ─ cdt.set_segmentation(segmentation)
    ─ cdt.generate_save(task='chord')
    ─ 輸出 chord_gen/
                       │
                       ▼
[7] TPQ 對齊 + Melody 重新量化
    ─ align_chord_gen_tpq()         — 讓 chord_gen 的 TPQ 與 processed_melody 一致
    ─ requantize_chord_gen_melody() — 修正 chorderator 在 melody track 上的 rounding error
                       │
                       ▼
[8] 填補空小節
    ─ fill_empty_bars_with_chords(processed_melody, chord_gen, empty_bars)
    ─ 輸出 chord_gen_filled_empty/
                       │
                       ▼
[9] 量化到 1/16
    ─ quantize_melody_to_16th()
    ─ 輸出 chord_gen_quantized/
                       │
                       ▼
[10] 輸出和弦文字檔
    ─ export_chords_txt_chorder(filled_output, beat_times=beat_times_aligned)
      ← 傳 beat_times_aligned（從 downbeat 開始），
        確保 chord beat 0 對應到 bar 1 beat 1 的真實秒數
    ─ fill_none_chords_in_txt()  — 用前後和弦填補 None
    ─ 輸出 chord_txt_with_None/ 和 chord_txt/
                       │
                       ▼
最終輸出：每行 onset, offset, chord_symbol 的 .txt 檔
（時間為真實音頻秒數，從第一個 downbeat 開始對齊）
```

---

## 已知限制

| 限制 | 說明 |
|------|------|
| CDT 只支援單一 tempo | 整首歌用平均 BPM，tempo 變化大的段落仍有誤差（future work） |
| CDT 只支援 4/4、4-bar/8-bar phrase | 其他拍號或非整數 phrase 會被強制 round |
| Pickup notes 被丟棄 | warp 後早於 downbeat_phase 的音符會被推到 bar 0，視為無效 |
| downbeat_phase 依賴音符分布 | 若 vocal 旋律非常不規則（無主要強拍）可能偵測錯誤 |

---

## 相關函式對照

| 函式 | 位置 | 說明 |
|------|------|------|
| `load_beat_times()` | `demo_utils.py:14` | 讀 SingNet 格式的 beat_times.txt |
| `estimate_tempo_from_beats()` | `demo_utils.py:36` | 從 beat_times 算平均 BPM（含 octave correction） |
| `detect_downbeat_phase()` | `demo_utils.py:460` | 從音符 onset 分布推斷 downbeat phase |
| `warp_midi_to_beats()` | `demo_utils.py:227` | Beat-aware warping + 量化到 beat grid |
| `get_auto_config()` | `demo_utils.py:1340` | 從 MIDI 算 segmentation / note_shift |
| `align_chord_gen_tpq()` | `demo_utils.py:2476` | 對齊 chord_gen 與 melody 的 TPQ |
| `requantize_chord_gen_melody()` | `demo_utils.py:2607` | 修正 melody track 的 rounding error |
| `export_chords_txt_chorder()` | `demo_utils.py:1710` | 用 Dechorder 輸出 onset/offset/chord txt |
| `fill_none_chords_in_txt()` | `demo_utils.py:1819` | 填補 None 和弦 |
