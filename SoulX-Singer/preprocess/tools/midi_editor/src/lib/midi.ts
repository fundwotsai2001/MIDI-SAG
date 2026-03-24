import { Midi } from '@tonejs/midi'
import { writeMidi } from 'midi-file'
import type { MidiData, MidiEvent } from 'midi-file'
import type { NoteEvent, ProjectSnapshot, TimeSignature } from '../types'

const DEFAULT_SIGNATURE: TimeSignature = [4, 4]

// Decode UTF-8 byte string (latin1 encoded) to proper Unicode string
// This matches: text.encode("latin1").decode("utf-8") in Python
function decodeUtf8ByteString(byteString: string): string {
  try {
    const bytes = new Uint8Array(byteString.length)
    for (let i = 0; i < byteString.length; i++) {
      bytes[i] = byteString.charCodeAt(i)
    }
    return new TextDecoder('utf-8').decode(bytes)
  } catch {
    return byteString
  }
}

// Encode Unicode string to UTF-8 byte string (latin1 encoding)
// This matches: text.encode("utf-8").decode("latin1") in Python
function encodeUtf8ByteString(text: string): string {
  const bytes = new TextEncoder().encode(text)
  let output = ''
  bytes.forEach((b) => {
    output += String.fromCharCode(b)
  })
  return output
}

export async function importMidiFile(file: File): Promise<ProjectSnapshot> {
  const buffer = await file.arrayBuffer()
  return parseMidiBuffer(buffer)
}

export async function parseMidiBuffer(buffer: ArrayBuffer): Promise<ProjectSnapshot> {
  const midi = new Midi(buffer)
  const tempo = midi.header.tempos[0]?.bpm ?? 120
  const timeSignature = (midi.header.timeSignatures[0]?.timeSignature as TimeSignature | undefined) ?? DEFAULT_SIGNATURE
  
  // Merge notes from all tracks and sort by ticks then by midi (for stable ordering)
  const allNotes = midi.tracks
    .flatMap(t => t.notes)
    .sort((a, b) => a.ticks - b.ticks || a.midi - b.midi)
  
  // Get lyrics from header.meta and sort by ticks
  const lyricEvents = midi.header.meta
    .filter((event) => event.type === 'lyrics')
    .sort((a, b) => a.ticks - b.ticks)
  
  // Match lyrics to notes by tick position
  // Each lyric should be consumed by exactly one note at the same tick
  const lyricsByTick = new Map<number, string[]>()
  for (const event of lyricEvents) {
    const existing = lyricsByTick.get(event.ticks) || []
    existing.push(decodeUtf8ByteString(event.text))
    lyricsByTick.set(event.ticks, existing)
  }
  
  // Track which lyrics have been used at each tick position
  const usedLyricIndices = new Map<number, number>()
  
  const notes: NoteEvent[] = allNotes.map((note, index) => {
    const beat = note.ticks / midi.header.ppq
    const durationBeats = note.durationTicks / midi.header.ppq
    
    let lyric = ''
    
    // First try exact tick match
    const lyricsAtTick = lyricsByTick.get(note.ticks)
    if (lyricsAtTick && lyricsAtTick.length > 0) {
      const usedIndex = usedLyricIndices.get(note.ticks) || 0
      if (usedIndex < lyricsAtTick.length) {
        lyric = lyricsAtTick[usedIndex]
        usedLyricIndices.set(note.ticks, usedIndex + 1)
      }
    }
    
    // If no exact match, try nearby ticks (within small tolerance)
    if (!lyric) {
      const tolerance = midi.header.ppq / 100 // Very small tolerance
      for (const [tick, lyrics] of lyricsByTick.entries()) {
        if (Math.abs(tick - note.ticks) <= tolerance) {
          const usedIndex = usedLyricIndices.get(tick) || 0
          if (usedIndex < lyrics.length) {
            lyric = lyrics[usedIndex]
            usedLyricIndices.set(tick, usedIndex + 1)
            break
          }
        }
      }
    }

    return {
      id: `${index}-${note.midi}-${Math.round(note.ticks)}`,
      midi: note.midi,
      start: beat,
      duration: Math.max(durationBeats, 0.0625),
      velocity: note.velocity,
      lyric,
    }
  })

  return { tempo, timeSignature, notes, ppq: midi.header.ppq }
}

// Used to add absoluteTime property for sorting
type WithAbsoluteTime<T> = T & { absoluteTime: number }

export function exportMidi(snapshot: ProjectSnapshot): Blob {
  const ppq = snapshot.ppq ?? 480  // Use original ppq if available, otherwise default to 480
  const microsecondsPerBeat = Math.round(60000000 / snapshot.tempo)  // Convert BPM to microseconds per beat

  // Sort notes by start time, then by midi for stable ordering
  const sortedNotes = [...snapshot.notes].sort((a, b) => a.start - b.start || a.midi - b.midi)

  // Build events for a single track containing both lyrics and notes
  // Event order at same tick: note_off (0) < lyrics (1) < note_on (2)
  // This matches meta.py's tg2midi implementation
  const events: Array<WithAbsoluteTime<MidiEvent>> = []

  // Add all note events and their corresponding lyrics
  sortedNotes.forEach((note) => {
    const startTicks = Math.round(note.start * ppq)
    const endTicks = Math.round((note.start + note.duration) * ppq)
    const velocity = Math.round(note.velocity * 127)

    // Add lyric event at the same tick as note_on (but will be sorted before it)
    const lyricText = note.lyric ?? ''
    const encodedLyric = encodeUtf8ByteString(lyricText)
    
    // Lyric event - sort key 1 (after note_off, before note_on)
    events.push({
      absoluteTime: startTicks,
      deltaTime: 0,
      meta: true,
      type: 'lyrics',
      text: encodedLyric,
      _sortKey: 1,
    } as WithAbsoluteTime<MidiEvent> & { _sortKey: number })

    // Note on event - sort key 2 (after lyrics)
    events.push({
      absoluteTime: startTicks,
      deltaTime: 0,
      type: 'noteOn',
      channel: 0,
      noteNumber: note.midi,
      velocity: velocity,
      _sortKey: 2,
    } as WithAbsoluteTime<MidiEvent> & { _sortKey: number })

    // Note off event - sort key 0 (before everything at same tick)
    events.push({
      absoluteTime: endTicks,
      deltaTime: 0,
      type: 'noteOff',
      channel: 0,
      noteNumber: note.midi,
      velocity: 0,
      _sortKey: 0,
    } as WithAbsoluteTime<MidiEvent> & { _sortKey: number })
  })

  // Sort events by absoluteTime, then by _sortKey
  events.sort((a, b) => {
    const aKey = (a as { _sortKey?: number })._sortKey ?? 1
    const bKey = (b as { _sortKey?: number })._sortKey ?? 1
    return a.absoluteTime - b.absoluteTime || aKey - bKey
  })

  // Convert absolute time to delta time
  let lastTick = 0
  events.forEach(event => {
    event.deltaTime = event.absoluteTime - lastTick
    lastTick = event.absoluteTime
    delete (event as { absoluteTime?: number }).absoluteTime
    delete (event as { _sortKey?: number })._sortKey
  })

  // Build the MIDI track with header events
  const track: MidiEvent[] = [
    // Set tempo
    {
      deltaTime: 0,
      meta: true,
      type: 'setTempo',
      microsecondsPerBeat: microsecondsPerBeat,
    },
    // Time signature
    {
      deltaTime: 0,
      meta: true,
      type: 'timeSignature',
      numerator: snapshot.timeSignature[0],
      denominator: snapshot.timeSignature[1],
      metronome: 24,
      thirtyseconds: 8,
    },
    // All note and lyric events
    ...events,
    // End of track
    {
      deltaTime: 0,
      meta: true,
      type: 'endOfTrack',
    },
  ]

  // Build MIDI data structure
  const midiData: MidiData = {
    header: {
      format: 0,  // Single track format (type 0)
      numTracks: 1,
      ticksPerBeat: ppq,
    },
    tracks: [track],
  }

  const bytes = writeMidi(midiData)
  return new Blob([new Uint8Array(bytes)], { type: 'audio/midi' })
}
