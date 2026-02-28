from pretty_midi import PrettyMIDI, Instrument, Note

from ...utils.excp import handle_exception
from ...utils.utils import MIDILoader
from ...utils.structured import major_map, minor_map, str_to_root


class PreProcessor:
    # Extended to support all multiples of 4 up to 64
    accepted_phrase_length = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]

    def __init__(self, midi_path='', phrase=None, meta=None, note_shift=0, **kwargs):
        self.midi_path = None
        self.note_shift = note_shift
        self.midi = PrettyMIDI(midi_path)
        self.melo = self.midi.instruments[0]
        self.meta = meta
        self.phrase = phrase

    def get(self):

        if self.midi_path is not None:
            pop909_loader = MIDILoader(files='POP909')
            pop909_loader.config(output_form='number')
            melo_source_name = MIDILoader.auto_find_pop909_source_name(start_with=self.midi_path)[:5]
            splited_melo = [pop909_loader.get(name=i) for i in melo_source_name]
            self.meta['pos'] = [name[6] for name in melo_source_name]
            self.meta['tempo'] = 120
            self.meta['unit'] = 0.125
        else:
            # TODO
            splited_melo = self.__analyze_midi()
            self.meta['pos'] = ['x' for i in splited_melo]
            if 'tempo' not in self.meta.keys():
                self.meta['tempo'] = self.midi.get_tempo_changes()[1][0]
            self.meta['unit'] = 60 / self.meta['tempo'] / 4
            print(self.meta)

        for i in splited_melo:
            phrase_bars = len(i) // 16
            print(f"Phrase length: {len(i)} positions = {phrase_bars} bars")
            # Accept any multiple of 4 (minimum 4 bars)
            if phrase_bars < 4 or phrase_bars % 4 != 0:
                handle_exception(312)
        # print(splited_melo)
        return self.melo, splited_melo, self.meta

    def __load_pop909_melo(self):
        pop909_loader = MIDILoader(files='POP909')
        pitch_list = pop909_loader.get_full_midi_ins_from_pop909(index=self.midi_path, change_key_to='C')
        ins = Instrument(program=0)
        current_pitch = pitch_list[0]
        start = 0
        for i in range(len(pitch_list)):
            if i == len(pitch_list) - 1:
                note = Note(pitch=current_pitch, velocity=80, start=start * 0.125, end=(i + 1) * 0.125)
                ins.notes.append(note)
                break
            if pitch_list[i + 1] != pitch_list[i]:
                if current_pitch != 0:
                    note = Note(pitch=current_pitch, velocity=80, start=start * 0.125, end=(i + 1) * 0.125)
                    ins.notes.append(note)
                current_pitch = pitch_list[i + 1]
                start = i + 1
        return ins

    def __analyze_midi(self):

        def quantize_note(time, unit):
            base = time // unit
            # print(f"time: {time}, unit: {unit}, base: {base}")
            return base if time % unit < unit / 2 else base + 1

        def pitch_to_number(pitch, meta):
            tonic_distance = str_to_root[meta['tonic']]
            if meta['mode'] == 'maj':
                return major_map[(pitch - tonic_distance) % 12]
            else:
                return minor_map[(pitch - tonic_distance) % 12]

        all_notes_and_pos = []

        if 'tempo' not in self.meta.keys():
            # unit = 60 / self.midi.estimate_tempo() / 4
            unit = 60 / self.midi.get_tempo_changes()[1][0] / 4
        else:
            unit = 60 / self.meta['tempo'] / 4
        for note in self.midi.instruments[0].notes:
            if quantize_note(note.end, unit) >= self.note_shift:
                all_notes_and_pos.append([quantize_note(note.start, unit)-self.note_shift,
                                          quantize_note(note.end, unit)-self.note_shift,
                                          pitch_to_number(note.pitch, self.meta),
                                          note.velocity])

        melo_sequence = self.__construct_melo_sequence(all_notes_and_pos)
        splited_melo = []
        print(len(melo_sequence))

        if self.phrase[0] != 1:
            melo_sequence = melo_sequence[16 * (self.phrase[0] - 1):]
        for i in range(len(self.phrase)):
            if i == 0:
                continue
            splited_melo.append(melo_sequence[16 * (self.phrase[i - 1] - 1):16 * (self.phrase[i] - 1)])
            if i == len(self.phrase) - 1:
                splited_melo.append(melo_sequence[16 * (self.phrase[i] - 1):])
                break

        if len(splited_melo) == 0:
            splited_melo = [melo_sequence]
        return splited_melo

    @staticmethod
    def __construct_melo_sequence(all_notes_and_pos):

        def fix_end(max_end):
            return int(max_end) if int(max_end) % 4 == 0 else int(((max_end//4)+1) * 4)

        def is_note_playing_at_cursor(note, cursor):
            return True if note[0] <= cursor < note[1] else False

        # print(all_notes_and_pos)
        max_end = max(all_notes_and_pos, key=lambda x: x[1])[1]
        print(f"max_end: {max_end}")

        fixed_end = fix_end(max_end // 16)

        if fixed_end is None:
            handle_exception(312)
        else:
            fixed_end *= 16
        print(f"fixed_end: {fixed_end}")

        # fixed_end = int((max_end // 16) * 16)

        note_dict = {i: all_notes_and_pos[i] for i in range(len(all_notes_and_pos))}
        melo_sequence, cache = [], -1
        for cursor in range(fixed_end):
            if cache != -1:
                if is_note_playing_at_cursor(note_dict[cache], cursor):
                    melo_sequence.append(note_dict[cache][2])
                    continue
                else:
                    cache = -1
            for (key, note) in note_dict.items():
                if is_note_playing_at_cursor(note, cursor):
                    melo_sequence.append(note[2])
                    cache = key
                    break
            else:
                melo_sequence.append(0)
        return melo_sequence


if __name__ == '__main__':
    midi_path = '/Users/billyyi/PycharmProjects/Chorderator/MIDI demos/inputs/6.mid'
    meta = {
        'tonic': 'C',
        'mode': 'maj',
        'meter': '4/4'
    }
    phrase = [1]
    pp = PreProcessor(midi_path=midi_path, phrase=phrase, meta=meta)
