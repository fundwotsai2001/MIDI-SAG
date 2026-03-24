import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import * as Tone from 'tone'
import { PianoRoll } from './components/PianoRoll'
import { LyricTable } from './components/LyricTable'
import { AudioTrack } from './components/AudioTrack'
import { useMidiStore } from './store/useMidiStore'
import { exportMidi, importMidiFile } from './lib/midi'
import type { TimeSignature } from './types'
import type { Lang } from './i18n'
import { getTranslations } from './i18n'
import { BASE_GRID_SECOND_WIDTH, BASE_ROW_HEIGHT, LOW_NOTE, HIGH_NOTE } from './constants'
import './App.css'

type PlayEvent = {
  time: number
  midi: number
  duration: number
  velocity: number
}

function App() {
  const {
    notes,
    tempo,
    timeSignature,
    selectedId,
    playhead,
    ppq,
    addNote,
    updateNote,
    removeNote,
    setNotes,
    setTempo,
    setTimeSignature,
    setPpq,
    select,
    setPlayhead,
  } = useMidiStore()

  const [lang, setLang] = useState<Lang>('zh')
  const t = getTranslations(lang)

  const [status, setStatus] = useState(t.ready)
  const [isPlaying, setIsPlaying] = useState(false)
  const [theme, setTheme] = useState<'dark' | 'light'>('light')
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [audioDuration, setAudioDuration] = useState(0)
  const [midiVolume, setMidiVolume] = useState(80) // 0-100
  const [audioVolume, setAudioVolume] = useState(80) // 0-100
  const [horizontalZoom, setHorizontalZoom] = useState(1)
  const [verticalZoom, setVerticalZoom] = useState(1)
  const [focusLyricId, setFocusLyricId] = useState<string | null>(null)
  // Selection range for loop playback (in seconds)
  const [selectionStart, setSelectionStart] = useState<number | null>(null)
  const [selectionEnd, setSelectionEnd] = useState<number | null>(null)
  const [isSelectingRange, setIsSelectingRange] = useState(false)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const audioInputRef = useRef<HTMLInputElement | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const partRef = useRef<Tone.Part<PlayEvent> | null>(null)
  const synthRef = useRef<Tone.PolySynth | null>(null)
  const rafRef = useRef<number | null>(null)
  const audioScrollRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    return () => {
      stopPlayback()
      synthRef.current?.dispose()
    }
  }, [])

  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])

  // Update status text when language changes
  useEffect(() => {
    setStatus(t.ready)
  }, [lang])

  // Sync audio volume - also trigger when audioUrl changes (new audio loaded)
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = audioVolume / 100
    }
  }, [audioVolume, audioUrl])

  // Sync MIDI synth volume
  useEffect(() => {
    if (synthRef.current) {
      // Convert 0-100 to dB scale (-60 to 0)
      const dbValue = midiVolume === 0 ? -Infinity : (midiVolume / 100) * 60 - 60
      synthRef.current.volume.value = dbValue
    }
  }, [midiVolume])

  useEffect(() => {
    if (!audioUrl) return
    return () => {
      URL.revokeObjectURL(audioUrl)
    }
  }, [audioUrl])

  const ensureSynth = async () => {
    await Tone.start()
    if (!synthRef.current) {
      synthRef.current = new Tone.PolySynth(Tone.Synth).toDestination()
      // Apply current volume
      const dbValue = midiVolume === 0 ? -Infinity : (midiVolume / 100) * 60 - 60
      synthRef.current.volume.value = dbValue
    }
  }

  const playPreviewNote = useCallback(async (midi: number) => {
    await ensureSynth()
    const frequency = Tone.Frequency(midi, 'midi').toFrequency()
    synthRef.current?.triggerAttackRelease(frequency, '8n', Tone.now(), 0.7)
  }, [midiVolume])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!selectedId) return
      const target = event.target as HTMLElement | null
      if (target && ['INPUT', 'TEXTAREA'].includes(target.tagName)) return
      
      // Delete note
      if (event.key === 'Backspace' || event.key === 'Delete') {
        event.preventDefault()
        removeNote(selectedId)
        select(null)
        return
      }
      
      // Cmd/Ctrl + Up/Down to adjust pitch
      const isCmdOrCtrl = event.metaKey || event.ctrlKey
      if (isCmdOrCtrl && (event.key === 'ArrowUp' || event.key === 'ArrowDown')) {
        event.preventDefault()
        const selectedNote = notes.find(n => n.id === selectedId)
        if (!selectedNote) return
        
        const delta = event.key === 'ArrowUp' ? 1 : -1
        const newMidi = Math.max(LOW_NOTE, Math.min(HIGH_NOTE, selectedNote.midi + delta))
        
        if (newMidi !== selectedNote.midi) {
          updateNote(selectedId, { midi: newMidi })
          playPreviewNote(newMidi)
        }
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [selectedId, notes, removeNote, select, updateNote, playPreviewNote])

  const noteEvents = useMemo<PlayEvent[]>(
    () =>
      notes.map((note) => ({
        time: (60 / tempo) * note.start,
        duration: (60 / tempo) * note.duration,
        midi: note.midi,
        velocity: note.velocity,
      })),
    [notes, tempo],
  )

  const beatToSeconds = (beat: number) => beat * (60 / tempo)
  const secondsToBeat = (seconds: number) => seconds / (60 / tempo)
  const seekBySeconds = (deltaSeconds: number) => {
    const maxNoteEnd = notes.reduce((acc, n) => Math.max(acc, n.start + n.duration), 0)
    const maxBeat = Math.max(secondsToBeat(audioDuration), maxNoteEnd)
    const nextSeconds = Math.max(0, Math.min(beatToSeconds(maxBeat), beatToSeconds(playhead) + deltaSeconds))
    seekToBeat(secondsToBeat(nextSeconds))
  }

  const gridSecondWidth = BASE_GRID_SECOND_WIDTH * horizontalZoom
  const rowHeight = BASE_ROW_HEIGHT * verticalZoom

  // Calculate MIDI content width to sync with audio track
  const midiContentWidth = useMemo(() => {
    const noteEndSeconds = notes.reduce((acc, n) => {
      const endBeat = n.start + n.duration
      return Math.max(acc, beatToSeconds(endBeat))
    }, 8)
    const maxSeconds = Math.max(noteEndSeconds + 10, audioDuration + 10, 30)
    return maxSeconds * gridSecondWidth
  }, [notes, audioDuration, gridSecondWidth, beatToSeconds])

  const seekToBeat = (beat: number) => {
    setPlayhead(beat)
    Tone.Transport.seconds = beatToSeconds(beat)
    if (audioRef.current) {
      audioRef.current.currentTime = beatToSeconds(beat)
    }
  }

  const schedulePlayback = async () => {
    if (!notes.length && !audioUrl) return
    await ensureSynth()
    partRef.current?.dispose()
    Tone.Transport.cancel()
    Tone.Transport.stop()
    Tone.Transport.bpm.value = tempo

    // Determine playback range
    const hasSelection = selectionStart !== null && selectionEnd !== null && selectionEnd > selectionStart
    const startSeconds = hasSelection ? selectionStart : beatToSeconds(playhead)
    const endSeconds = hasSelection ? selectionEnd : null

    Tone.Transport.seconds = startSeconds

    // Filter notes within selection range if applicable
    const filteredEvents = hasSelection
      ? noteEvents.filter(e => e.time >= startSeconds && e.time < endSeconds!)
      : noteEvents

    if (filteredEvents.length) {
      partRef.current = new Tone.Part((time, event) => {
        if (midiVolume === 0) return
        const frequency = Tone.Frequency(event.midi, 'midi').toFrequency()
        synthRef.current?.triggerAttackRelease(frequency, event.duration, time, event.velocity)
      }, filteredEvents)
      partRef.current.start(0)
    }
    Tone.Transport.start()
    if (audioRef.current && audioUrl) {
      audioRef.current.currentTime = startSeconds
      if (audioVolume > 0) {
        audioRef.current.play().catch(() => null)
      }
    }
    setIsPlaying(true)
    setStatus(hasSelection ? t.selectionPlayback : t.playing)

    const tick = () => {
      const seconds =
        audioRef.current && audioUrl && !audioRef.current.paused
          ? audioRef.current.currentTime
          : Tone.Transport.seconds
      
      // Stop at selection end
      if (endSeconds !== null && seconds >= endSeconds) {
        pausePlayback()
        seekToBeat(secondsToBeat(selectionStart!))
        setStatus(t.selectionDone)
        return
      }
      
      const beat = seconds / (60 / tempo)
      setPlayhead(beat)
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
  }

  const stopPlayback = () => {
    Tone.Transport.stop()
    Tone.Transport.cancel()
    partRef.current?.dispose()
    partRef.current = null
    setIsPlaying(false)
    setPlayhead(0)
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0
    }
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
  }

  const pausePlayback = () => {
    Tone.Transport.stop()
    partRef.current?.dispose()
    partRef.current = null
    setIsPlaying(false)
    if (audioRef.current) {
      audioRef.current.pause()
    }
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
  }

  const handlePlayToggle = async () => {
    if (isPlaying) {
      pausePlayback()
      setStatus(t.paused)
    } else {
      await schedulePlayback()
    }
  }

  const handleImportClick = () => fileInputRef.current?.click()
  const handleAudioImportClick = () => audioInputRef.current?.click()

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    try {
      const snapshot = await importMidiFile(file)
      setNotes(snapshot.notes)
      setTempo(snapshot.tempo)
      setTimeSignature(snapshot.timeSignature as TimeSignature)
      setPpq(snapshot.ppq)  // Preserve original ppq for accurate export
      setStatus(t.imported(file.name))
    } catch (error) {
      console.error(error)
      setStatus(t.importFailed)
    } finally {
      event.target.value = ''
    }
  }

  const handleAudioChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    
    // Validate audio file type
    const validAudioTypes = ['audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/flac', 'audio/mp4', 'audio/aac', 'audio/x-m4a']
    const validExtensions = ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac']
    const fileName = file.name.toLowerCase()
    const isValidType = validAudioTypes.includes(file.type) || file.type.startsWith('audio/')
    const isValidExtension = validExtensions.some(ext => fileName.endsWith(ext))
    
    if (!isValidType && !isValidExtension) {
      setStatus(t.unsupportedFormat(validExtensions.join(', ')))
      event.target.value = ''
      return
    }
    
    const url = URL.createObjectURL(file)
    setAudioUrl(url)
    setStatus(t.audioImported(file.name))
    event.target.value = ''
  }

  // Fix overlapping notes by trimming the first note to end where the second begins
  // Returns the number of fixed overlaps
  const fixOverlaps = (): number => {
    const sortedNotes = [...notes].sort((a, b) => a.start - b.start)
    let fixCount = 0
    
    for (let i = 0; i < sortedNotes.length - 1; i++) {
      const noteA = sortedNotes[i]
      const noteB = sortedNotes[i + 1]
      const noteAEnd = noteA.start + noteA.duration
      
      // If noteA overlaps with noteB
      if (noteAEnd > noteB.start) {
        // Trim noteA to end at noteB's start
        const newDuration = Math.max(0.01, noteB.start - noteA.start)
        updateNote(noteA.id, { duration: newDuration })
        fixCount++
      }
    }
    
    return fixCount
  }

  // UI handler for fix overlaps button
  const handleFixOverlaps = () => {
    const fixCount = fixOverlaps()
    if (fixCount > 0) {
      setStatus(t.fixedOverlaps(fixCount))
    } else {
      setStatus(t.noOverlaps)
    }
  }

  const handleExport = () => {
    // Auto-fix overlaps before export
    fixOverlaps()
    
    // Get the latest notes from store (after fix, zustand set is synchronous)
    const latestNotes = useMidiStore.getState().notes
    
    const blob = exportMidi({ notes: latestNotes, tempo, timeSignature, ppq })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'vocal-midi.mid'
    anchor.click()
    URL.revokeObjectURL(url)
    setStatus(t.exported)
  }

  const handleTranspose = (semitones: number) => {
    if (semitones === 0 || !notes.length) return
    for (const note of notes) {
      const newMidi = Math.max(0, Math.min(127, note.midi + semitones))
      updateNote(note.id, { midi: newMidi })
    }
    setStatus(t.transposed(semitones))
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">{t.eyebrow}</p>
          <h1>{t.title}</h1>
          <p className="muted">{t.subtitle}</p>
        </div>
        <div className="actions">
          <button className="primary" onClick={handleImportClick}>
            {t.importMidi}
          </button>
          <button className="primary" onClick={handleExport}>
            {t.exportMidi}
          </button>
          <div className="transpose-group" title={t.transposeTooltip}>
            <select
              className="transpose-select"
              value={0}
              onChange={(e) => {
                const val = Number(e.target.value)
                if (val !== 0) handleTranspose(val)
                e.target.value = '0'
              }}
            >
              <option value={0}>{t.transpose}</option>
              {Array.from({ length: 24 }, (_, i) => i - 12)
                .filter(v => v !== 0)
                .reverse()
                .map(v => (
                  <option key={v} value={v}>
                    {v > 0 ? `+${v}` : v}
                  </option>
                ))}
            </select>
          </div>
          <button className="soft" onClick={handleFixOverlaps} title={t.fixOverlapsTooltip}>
            {t.fixOverlaps}
          </button>
          <button className="icon-toggle" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>
            {theme === 'dark' ? (
              <span className="icon" aria-label={t.switchToLight}>
                ☀️
              </span>
            ) : (
              <span className="icon" aria-label={t.switchToDark}>
                🌙
              </span>
            )}
          </button>
          <button
            className="icon-toggle"
            onClick={() => setLang(lang === 'zh' ? 'en' : 'zh')}
            title={lang === 'zh' ? 'Switch to English' : '切换到中文'}
          >
            <span className="lang-label">{lang === 'zh' ? 'EN' : '中'}</span>
          </button>
          <input ref={fileInputRef} type="file" accept=".mid,.midi" className="sr-only" onChange={handleFileChange} />
        </div>
      </header>

      <section className="audio-bar">
        <div className="audio-left">
          <button className="ghost" onClick={handleAudioImportClick}>
            {t.importAudio}
          </button>
          <input
            ref={audioInputRef}
            type="file"
            accept=".mp3,.wav,.ogg,.flac,.m4a,.aac"
            className="sr-only"
            onChange={handleAudioChange}
          />
          <span className="audio-hint">{t.audioHint}</span>
        </div>
        <div className="audio-right">
          <div className="volume-control">
            <span className="volume-label">{t.midiLabel}</span>
            <input
              type="range"
              min={0}
              max={100}
              value={midiVolume}
              onChange={(e) => setMidiVolume(Number(e.target.value))}
              className="volume-slider"
            />
            <span className="volume-value">{midiVolume}%</span>
          </div>
          <div className="volume-control">
            <span className="volume-label">{t.audioLabel}</span>
            <input
              type="range"
              min={0}
              max={100}
              value={audioVolume}
              onChange={(e) => setAudioVolume(Number(e.target.value))}
              className="volume-slider"
            />
            <span className="volume-value">{audioVolume}%</span>
          </div>
        </div>
      </section>

      <section className="panel panel-split">
        <div className="panel-main">
          {audioUrl && (
            <AudioTrack
              key={audioUrl}
              ref={audioScrollRef}
              audioUrl={audioUrl}
              muted={audioVolume === 0}
              onSeek={(seconds) => seekToBeat(secondsToBeat(seconds))}
              playheadSeconds={beatToSeconds(playhead)}
              gridSecondWidth={gridSecondWidth}
              minContentWidth={midiContentWidth}
            />
          )}
          <PianoRoll
            notes={notes}
            selectedId={selectedId}
            timeSignature={timeSignature}
            tempo={tempo}
            playhead={playhead}
            selectionStart={selectionStart}
            selectionEnd={selectionEnd}
            onAddNote={addNote}
            onSelect={select}
            onUpdateNote={updateNote}
            onSeek={seekToBeat}
            onScroll={(left) => {
              if (audioScrollRef.current) {
                audioScrollRef.current.scrollLeft = left
              }
            }}
            onZoom={(deltaH, deltaV) => {
              if (deltaH !== 0) {
                setHorizontalZoom(prev => Math.max(0.5, prev + deltaH))
              }
              if (deltaV !== 0) {
                setVerticalZoom(prev => Math.max(0.6, Math.min(2.5, prev + deltaV)))
              }
            }}
            onPlayNote={playPreviewNote}
            onFocusLyric={(noteId) => {
              select(noteId)
              setFocusLyricId(noteId)
            }}
            onSelectionChange={(start, end) => {
              setSelectionStart(start)
              setSelectionEnd(end)
            }}
            isSelectingRange={isSelectingRange}
            audioDuration={audioDuration}
            gridSecondWidth={gridSecondWidth}
            rowHeight={rowHeight}
          />
        </div>
        <aside className="panel-side">
          <div className="controls">
            <div className="toggle" style={{ justifyContent: 'space-between' }}>
              <span>{t.horizontalZoom}</span>
              <input
                type="range"
                min={0.5}
                max={10}
                step={0.1}
                value={Math.min(horizontalZoom, 10)}
                onChange={(e) => setHorizontalZoom(Number(e.target.value))}
                style={{ width: '140px' }}
              />
              <span style={{ width: 42, textAlign: 'right' }}>{horizontalZoom.toFixed(1)}x</span>
            </div>
            <div className="toggle" style={{ justifyContent: 'space-between' }}>
              <span>{t.verticalZoom}</span>
              <input
                type="range"
                min={0.6}
                max={2.5}
                step={0.1}
                value={verticalZoom}
                onChange={(e) => setVerticalZoom(Number(e.target.value))}
                style={{ width: '140px' }}
              />
              <span style={{ width: 42, textAlign: 'right' }}>{verticalZoom.toFixed(1)}x</span>
            </div>
            <div className="transport">
              <button 
                className="soft" 
                onClick={() => {
                   setPlayhead(0)
                   seekToBeat(0)
                }}
                title={t.goToStart}
              >
                ⏮
              </button>
              <button
                className="soft"
                onClick={() => seekBySeconds(-2)}
                title={t.back2s}
              >
                ⏪
              </button>
              <button 
                 className="primary" 
                 onClick={handlePlayToggle} 
                 disabled={!notes.length && !audioUrl}
                 title={isPlaying ? t.pause : (selectionStart !== null && selectionEnd !== null ? t.playSelection : t.play)}
              >
                {isPlaying ? '⏸' : '▶'}
              </button>
              <button
                className="soft"
                onClick={() => seekBySeconds(2)}
                title={t.forward2s}
              >
                ⏩
              </button>
              <button 
                 className="soft"
                 onClick={() => {
                    const maxNoteEnd = notes.reduce((acc, n) => Math.max(acc, n.start + n.duration), 0)
                    seekToBeat(Math.max(secondsToBeat(audioDuration), maxNoteEnd))
                 }}
                 title={t.goToEnd}
              >
                ⏭
              </button>
              <span className="transport-divider" />
              <button
                className={`soft selection-btn ${isSelectingRange ? 'active' : ''}`}
                onClick={() => {
                  if (isSelectingRange) {
                    // Exiting selection mode - auto clear selection
                    setIsSelectingRange(false)
                    setSelectionStart(null)
                    setSelectionEnd(null)
                  } else {
                    setIsSelectingRange(true)
                  }
                }}
                title={isSelectingRange ? t.exitSelectMode : t.setRangeTooltip}
              >
                {isSelectingRange ? `📍 ${t.selectingRange}` : `📍 ${t.setRange}`}
              </button>
            </div>
            <div className="status">{status}</div>
          </div>
          <div className="lyric-container">
            <LyricTable 
              notes={notes} 
              selectedId={selectedId} 
              tempo={tempo} 
              focusLyricId={focusLyricId}
              lang={lang}
              onSelect={select} 
              onUpdate={updateNote}
              onFocusHandled={() => setFocusLyricId(null)}
            />
          </div>
        </aside>
      </section>
      <audio 
        ref={audioRef} 
        src={audioUrl ?? undefined} 
        preload="auto" 
        className="sr-only" 
        onLoadedMetadata={(e) => {
          setAudioDuration(e.currentTarget.duration)
          // Ensure volume is set when audio loads
          e.currentTarget.volume = audioVolume / 100
        }}
      />
    </div>
  )
}

export default App
