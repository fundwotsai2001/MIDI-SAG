export type Lang = 'zh' | 'en'

const zh = {
  // Header
  eyebrow: '歌声 MIDI 编辑器',
  title: 'SoulX-Singer MIDI Editor',
  subtitle: '导入、拖拽、实时修改歌词并导出标准 MIDI。',
  switchToLight: '切换到亮色',
  switchToDark: '切换到暗色',
  importJson: '导入 JSON',
  exportJson: '导出 JSON',
  importMidi: '导入 MIDI',
  exportMidi: '导出 MIDI',
  transpose: '移调',
  transposeTooltip: '整体升降调：所有音符的音高同步改变',
  transposed: (n: number) => `已移调 ${n > 0 ? '+' : ''}${n} 半音`,
  fixOverlaps: '消除重叠',
  fixOverlapsTooltip: '自动消除重叠：将重叠音符的音尾提前到下一个音的音头',
  jsonImported: (name: string) => `已从 JSON 载入 ${name}`,
  jsonImportFailed: 'JSON 导入失败，请确认文件格式正确',
  jsonExported: '已导出 META JSON 文件',

  // Audio bar
  importAudio: '对齐音频导入',
  audioHint: '导入后显示音频波形并与 MIDI 同步走带',
  midiLabel: 'MIDI',
  audioLabel: '音频',

  // Controls
  horizontalZoom: '水平缩放',
  verticalZoom: '垂直缩放',
  goToStart: '回到开头',
  back2s: '后退 2 秒',
  pause: '暂停',
  playSelection: '播放选区',
  play: '播放',
  forward2s: '前进 2 秒',
  goToEnd: '回到结尾',
  selectingRange: '选区中',
  setRange: '设选区',
  exitSelectMode: '退出选区模式（并清除选区）',
  setRangeTooltip: '设置选区：在时间轴上拖拽选择播放范围',

  // Status
  ready: '准备就绪',
  selectionPlayback: '选区回放中...',
  playing: '正在回放...',
  selectionDone: '选区播放完毕',
  paused: '已暂停',
  imported: (name: string) => `已载入 ${name}`,
  importFailed: '导入失败，请确认文件合法',
  audioImported: (name: string) => `已载入音频 ${name}`,
  unsupportedFormat: (exts: string) => `不支持的文件格式，请选择音频文件（${exts}）`,
  fixedOverlaps: (count: number) => `已修复 ${count} 个重叠音符`,
  noOverlaps: '没有检测到重叠音符',
  exported: '已导出包含歌词的 MIDI 文件',

  // Lyric table
  fillPlaceholderSelected: '从选中音符开始按词/字填充',
  fillPlaceholderDefault: '输入歌词，点击按词/字填充',
  fillButton: '按词\n填充',
  lyricPlaceholder: '输入歌词',
  emptyHint: '导入或双击钢琴卷帘以添加音符',
  confirmEdit: '确认修改 (Enter)',
}

const en: typeof zh = {
  // Header
  eyebrow: 'Vocal MIDI Editor',
  title: 'SoulX-Singer MIDI Editor',
  subtitle: 'Import, drag, edit lyrics in real-time, and export standard MIDI.',
  switchToLight: 'Switch to light',
  switchToDark: 'Switch to dark',
  importJson: 'Import JSON',
  exportJson: 'Export JSON',
  importMidi: 'Import MIDI',
  exportMidi: 'Export MIDI',
  transpose: 'Transpose',
  transposeTooltip: 'Transpose all notes up or down by semitones',
  transposed: (n: number) => `Transposed ${n > 0 ? '+' : ''}${n} semitone(s)`,
  fixOverlaps: 'Fix Overlaps',
  fixOverlapsTooltip: 'Auto fix overlaps: trim note end to the start of the next note',
  jsonImported: (name: string) => `Loaded from JSON ${name}`,
  jsonImportFailed: 'JSON import failed, please check the file format',
  jsonExported: 'Exported META JSON file',

  // Audio bar
  importAudio: 'Import Audio',
  audioHint: 'Display audio waveform synced with MIDI transport',
  midiLabel: 'MIDI',
  audioLabel: 'Audio',

  // Controls
  horizontalZoom: 'H-Zoom',
  verticalZoom: 'V-Zoom',
  goToStart: 'Go to start',
  back2s: 'Back 2s',
  pause: 'Pause',
  playSelection: 'Play selection',
  play: 'Play',
  forward2s: 'Forward 2s',
  goToEnd: 'Go to end',
  selectingRange: 'Selecting',
  setRange: 'Select',
  exitSelectMode: 'Exit selection mode (and clear selection)',
  setRangeTooltip: 'Set selection: drag on the timeline to select playback range',

  // Status
  ready: 'Ready',
  selectionPlayback: 'Playing selection...',
  playing: 'Playing...',
  selectionDone: 'Selection playback done',
  paused: 'Paused',
  imported: (name: string) => `Loaded ${name}`,
  importFailed: 'Import failed, please check the file',
  audioImported: (name: string) => `Loaded audio ${name}`,
  unsupportedFormat: (exts: string) => `Unsupported format, please select an audio file (${exts})`,
  fixedOverlaps: (count: number) => `Fixed ${count} overlapping note(s)`,
  noOverlaps: 'No overlapping notes detected',
  exported: 'Exported MIDI file with lyrics',

  // Lyric table
  fillPlaceholderSelected: 'Fill words from selected note',
  fillPlaceholderDefault: 'Enter lyrics, click fill button',
  fillButton: 'Fill\nWords',
  lyricPlaceholder: 'Type lyric',
  emptyHint: 'Import or double-click piano roll to add notes',
  confirmEdit: 'Confirm (Enter)',
}

const translations: Record<Lang, typeof zh> = { zh, en }

export type Translations = typeof zh

export function getTranslations(lang: Lang): Translations {
  return translations[lang]
}

// Smart tokenizer for lyrics: CJK characters are individual tokens, Latin words are grouped
function isCJK(char: string): boolean {
  const code = char.codePointAt(0) || 0
  return (
    (code >= 0x4E00 && code <= 0x9FFF) ||   // CJK Unified Ideographs
    (code >= 0x3400 && code <= 0x4DBF) ||   // CJK Extension A
    (code >= 0x20000 && code <= 0x2A6DF) || // CJK Extension B
    (code >= 0x3040 && code <= 0x309F) ||   // Hiragana
    (code >= 0x30A0 && code <= 0x30FF) ||   // Katakana
    (code >= 0xAC00 && code <= 0xD7AF)      // Hangul Syllables
  )
}

/**
 * Tokenize lyrics text for note filling.
 * - CJK characters: each character becomes one token (one per note)
 * - Latin/English words: each space-separated word becomes one token (one per note)
 * - Mixed text is handled correctly
 *
 * Examples:
 *   "你好世界"     -> ["你", "好", "世", "界"]
 *   "hello world"  -> ["hello", "world"]
 *   "I love 你"    -> ["I", "love", "你"]
 *   "something wrong" -> ["something", "wrong"]
 */
export function tokenizeLyrics(text: string): string[] {
  const tokens: string[] = []
  const cleaned = text.trim()
  if (!cleaned) return tokens

  let i = 0
  while (i < cleaned.length) {
    const char = cleaned[i]

    // Skip whitespace
    if (/\s/.test(char)) {
      i++
      continue
    }

    // CJK character - each is a separate token
    if (isCJK(char)) {
      tokens.push(char)
      i++
      continue
    }

    // Latin/number/other - collect until whitespace or CJK
    let word = ''
    while (i < cleaned.length && !/\s/.test(cleaned[i]) && !isCJK(cleaned[i])) {
      word += cleaned[i]
      i++
    }
    if (word) tokens.push(word)
  }

  return tokens
}
