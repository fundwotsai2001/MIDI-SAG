import { useEffect, useMemo, useRef, useState } from 'react'
import type { NoteEvent } from '../types'
import type { Lang } from '../i18n'
import { getTranslations, tokenizeLyrics } from '../i18n'

export type LyricTableProps = {
  notes: NoteEvent[]
  selectedId: string | null
  tempo: number
  focusLyricId: string | null
  lang: Lang
  onSelect: (id: string | null) => void
  onUpdate: (id: string, patch: Partial<NoteEvent>) => void
  onScrollToNote?: (noteId: string) => void
  onFocusHandled?: () => void
}

const formatSeconds = (beats: number, tempo: number) => {
  const seconds = beats * (60 / tempo)
  return Number.parseFloat(seconds.toFixed(2))
}

const secondsToBeats = (seconds: number, tempo: number) => {
  return seconds * (tempo / 60)
}

// Editable cell with confirmation
function EditableCell({ 
  value, 
  noteId,
  field,
  tempo,
  onConfirm,
  confirmTitle,
  type = 'number',
  min,
  step
}: {
  value: number
  noteId: string
  field: 'midi' | 'start' | 'end'
  tempo: number
  onConfirm: (noteId: string, field: string, value: number) => void
  confirmTitle?: string
  type?: string
  min?: number
  step?: number
}) {
  const displayValue = field === 'midi' ? value : formatSeconds(value, tempo)
  const [localValue, setLocalValue] = useState(String(displayValue))
  const [isDirty, setIsDirty] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  // Sync with external value when it changes (and not dirty)
  useEffect(() => {
    if (!isDirty) {
      setLocalValue(String(displayValue))
    }
  }, [displayValue, isDirty])

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setLocalValue(e.target.value)
    setIsDirty(true)
  }

  const handleConfirm = () => {
    const parsed = parseFloat(localValue)
    if (!isNaN(parsed)) {
      if (field === 'midi') {
        if (parsed >= 0 && parsed <= 127) {
          onConfirm(noteId, field, Math.round(parsed))
        }
      } else {
        if (parsed >= 0) {
          onConfirm(noteId, field, secondsToBeats(parsed, tempo))
        }
      }
    }
    setIsDirty(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      handleConfirm()
      inputRef.current?.blur()
    } else if (e.key === 'Escape') {
      setLocalValue(String(displayValue))
      setIsDirty(false)
      inputRef.current?.blur()
    }
  }

  const handleBlur = () => {
    if (isDirty) {
      // Reset to original on blur without confirm
      setLocalValue(String(displayValue))
      setIsDirty(false)
    }
  }

  return (
    <div className="editable-cell">
      <input
        ref={inputRef}
        className={`lyric-meta-input ${isDirty ? 'lyric-meta-dirty' : ''}`}
        type={type}
        min={min}
        step={step}
        value={localValue}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        onBlur={handleBlur}
        onClick={(e) => e.stopPropagation()}
      />
      {isDirty && (
        <button 
          className="confirm-btn"
          onMouseDown={(e) => {
            e.preventDefault() // Prevent input blur
            e.stopPropagation()
          }}
          onClick={(e) => {
            e.stopPropagation()
            handleConfirm()
          }}
          title={confirmTitle}
        >
          ✓
        </button>
      )}
    </div>
  )
}

export function LyricTable({ notes, selectedId, tempo, focusLyricId, lang, onSelect, onUpdate, onScrollToNote, onFocusHandled }: LyricTableProps) {
  const t = getTranslations(lang)
  const listRef = useRef<HTMLDivElement | null>(null)
  const inputRefs = useRef<Map<string, HTMLInputElement>>(new Map())
  const sorted = useMemo(() => [...notes].sort((a, b) => a.start - b.start), [notes])

  // Scroll to selected note (no auto-focus on single click)
  useEffect(() => {
    if (!selectedId || !listRef.current) return
    
    const target = listRef.current.querySelector<HTMLDivElement>(`[data-note-id="${selectedId}"]`)
    if (target) {
      target.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
    }
  }, [selectedId])

  // Focus lyric input when requested (double-click on note or click on list row)
  useEffect(() => {
    if (!focusLyricId) return
    
    const input = inputRefs.current.get(focusLyricId)
    if (input) {
      setTimeout(() => {
        input.focus()
        input.select()
      }, 50)
    }
    onFocusHandled?.()
  }, [focusLyricId, onFocusHandled])

  // Fill lyrics from selected note onwards
  // Uses smart tokenizer: CJK chars -> one per note, English words -> one per note
  const handleBulkFill = (bulkText: string) => {
    if (!sorted.length) return
    const tokens = tokenizeLyrics(bulkText)
    if (!tokens.length) return
    
    let startIndex = 0
    if (selectedId) {
      const selectedIndex = sorted.findIndex(n => n.id === selectedId)
      if (selectedIndex >= 0) {
        startIndex = selectedIndex
      }
    }
    
    let tokenIndex = 0
    for (let i = startIndex; i < sorted.length && tokenIndex < tokens.length; i++) {
      onUpdate(sorted[i].id, { lyric: tokens[tokenIndex] })
      tokenIndex++
    }
  }

  const handleRowClick = (noteId: string) => {
    onSelect(noteId)
    onScrollToNote?.(noteId)
  }

  const handleFieldConfirm = (noteId: string, field: string, value: number) => {
    const note = notes.find(n => n.id === noteId)
    if (!note) return

    if (field === 'midi') {
      onUpdate(noteId, { midi: value })
    } else if (field === 'start') {
      // Keep END the same, adjust duration accordingly
      const currentEnd = note.start + note.duration
      const newDuration = Math.max(0.01, currentEnd - value)
      onUpdate(noteId, { start: value, duration: newDuration })
    } else if (field === 'end') {
      // End changed, update duration
      const newDuration = Math.max(0.01, value - note.start)
      onUpdate(noteId, { duration: newDuration })
    }
  }

  return (
    <div className="lyric-card">
      <div className="lyric-bulk">
        <textarea
          className="lyric-bulk-input"
          rows={2}
          placeholder={selectedId ? t.fillPlaceholderSelected : t.fillPlaceholderDefault}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              handleBulkFill(e.currentTarget.value)
            }
          }}
        />
        <button
          className="soft"
          type="button"
          onClick={(e) => {
            const textarea = e.currentTarget.previousElementSibling as HTMLTextAreaElement
            handleBulkFill(textarea.value)
          }}
        >
          {t.fillButton.split('\n').map((line, i) => (
            <span key={i}>{line}{i === 0 && <br/>}</span>
          ))}
        </button>
      </div>
      <div className="lyric-header" style={{ flexShrink: 0 }}>
        <div>LYRIC</div>
        <div>PITCH</div>
        <div>START</div>
        <div>END</div>
      </div>
      <div className="lyric-list" ref={listRef}>
        {sorted.map((note) => (
          <div
            key={note.id}
            className={`lyric-row ${selectedId === note.id ? 'lyric-row-active' : ''}`}
            data-note-id={note.id}
            onClick={() => handleRowClick(note.id)}
          >
            <input
              ref={(el) => {
                if (el) {
                  inputRefs.current.set(note.id, el)
                } else {
                  inputRefs.current.delete(note.id)
                }
              }}
              className="lyric-input"
              value={note.lyric}
              placeholder={t.lyricPlaceholder}
              onChange={(event) => onUpdate(note.id, { lyric: event.target.value })}
              onClick={(e) => e.stopPropagation()}
            />
            <EditableCell
              value={note.midi}
              noteId={note.id}
              field="midi"
              tempo={tempo}
              onConfirm={handleFieldConfirm}
              confirmTitle={t.confirmEdit}
              min={0}
            />
            <EditableCell
              value={note.start}
              noteId={note.id}
              field="start"
              tempo={tempo}
              onConfirm={handleFieldConfirm}
              confirmTitle={t.confirmEdit}
              min={0}
              step={0.01}
            />
            <EditableCell
              value={note.start + note.duration}
              noteId={note.id}
              field="end"
              tempo={tempo}
              onConfirm={handleFieldConfirm}
              confirmTitle={t.confirmEdit}
              min={0}
              step={0.01}
            />
          </div>
        ))}
        {sorted.length === 0 && <div className="lyric-empty">{t.emptyHint}</div>}
      </div>
    </div>
  )
}
