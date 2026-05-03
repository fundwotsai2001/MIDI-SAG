import sys, os, random, time
from copy import deepcopy
sys.path.append('./model')

from model.CSLL2M import CSLL2M
from itertools import chain
from utils import pickle_load, numpy_to_tensor, tensor_to_numpy
from REMIaligned2midi import REMIaligned2midi

import torch
import yaml
import numpy as np
from scipy.stats import entropy

import jieba.posseg as pseg
import pypinyin
from pypinyin import Style

DEFAULT_BEAT_RESOL = 480
DEFAULT_BAR_RESOL = 480 * 4
DEFAULT_FRACTION = 64 

config_path = sys.argv[1]
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

device = config['training']['device']
data_dir = config['data']['data_dir']
#vocab_path = config['data']['vocab_path']
#data_split = config['data']['test_split']

ckpt_path = sys.argv[2]
out_dir = sys.argv[3]
n_samples_per_piece = int(sys.argv[4])
lyrics_path = sys.argv[5]
lyric2idx,idx2lyric = pickle_load(config['data']['vocab_path_lyric'])
event2idx,idx2event = pickle_load(config['data']['vocab_path_melody'])

from pypinyin import lazy_pinyin
# prepend_silent_bars.py
# midi_silence_bars.py
# midi_silence_bars_fixed.py
# midi_silence_bars_robust.py
# midi_silence_bars_conductor_safe.py
from mido import MidiFile, MidiTrack, MetaMessage, Message

ABS0_GLOBAL_META = {
    'time_signature', 'key_signature', 'set_tempo', 'smpte_offset',
    'track_name', 'marker', 'copyright', 'cue_marker'
}

def _find_time_signature(mid: MidiFile, use_last=True, default_ts=(4, 4)):
    """
    If use_last=True: return the last TS seen across all tracks by absolute time.
    Else: return a TS that occurs at absolute t==0 (or default if none).
    """
    if not use_last:
        # 't0' basis
        for track in mid.tracks:
            t = 0
            for msg in track:
                t += msg.time
                if t > 0:
                    break
                if msg.is_meta and msg.type == 'time_signature':
                    return (msg.numerator, msg.denominator)
        return default_ts

    # 'last' basis
    chosen, chosen_abs = None, -1
    for track in mid.tracks:
        t = 0
        for msg in track:
            t += msg.time
            if msg.is_meta and msg.type == 'time_signature':
                if t >= chosen_abs:
                    chosen, chosen_abs = (msg.numerator, msg.denominator), t
    return chosen if chosen else default_ts

def _ticks_per_bar(mid: MidiFile, use_last_ts=True, default_ts=(4, 4)) -> int:
    num, den = _find_time_signature(mid, use_last=use_last_ts, default_ts=default_ts)
    return int(round(mid.ticks_per_beat * num * (4 / den)))

def _last_tempo_mpqn(mid: MidiFile) -> int:
    """Return last tempo (microseconds per quarter note) seen across all tracks; default 500000."""
    mpqn, last_abs = 500000, -1
    for track in mid.tracks:
        t = 0
        for msg in track:
            t += msg.time
            if msg.is_meta and msg.type == 'set_tempo':
                if t >= last_abs:
                    mpqn, last_abs = msg.tempo, t
    return mpqn

def _find_conductor_track_idx(mid: MidiFile) -> int:
    """
    Heuristic: track that contains tempo or time_signature metas.
    If multiple, prefer the one that has any such meta at t==0.
    Fallback to track 0.
    """
    candidate, t0_candidate = None, None
    for idx, track in enumerate(mid.tracks):
        saw_meta = False
        t = 0
        for msg in track:
            t += msg.time
            if msg.is_meta and msg.type in ('set_tempo', 'time_signature'):
                saw_meta = True
                if t == 0 and t0_candidate is None:
                    t0_candidate = idx
                break  # good enough
            if t > 0:
                break
        if saw_meta and candidate is None:
            candidate = idx
    return t0_candidate if t0_candidate is not None else (candidate if candidate is not None else 0)

def add_silent_bars(in_path, out_path, bars=4, where='append',
                    default_ts=(4, 4),
                    prepend_use_t0_ts=True,
                    append_use_last_ts=True):
    """
    Add silent bars by either shifting content (prepend) or extending the timeline (append).
    - Prepend: shift first non-global message on every track by N bars.
    - Append : extend the conductor/tempo track by N bars with a duplicate tempo meta,
               and also add a padding track with a neutral channel event (belt & suspenders).
    """
    mid = MidiFile(in_path)

    if bars <= 0:
        mid.save(out_path)
        return

    if where not in ('prepend', 'append'):
        raise ValueError("where must be 'prepend' or 'append'")

    # --- PREPEND ---
    if where == 'prepend':
        delay = bars * _ticks_per_bar(mid, use_last_ts=not prepend_use_t0_ts, default_ts=default_ts)
        for track in mid.tracks:
            acc = 0
            insert_after = 0
            insert_idx = None
            for i, msg in enumerate(track):
                acc += msg.time
                if acc > 0:
                    insert_idx = i
                    break
                if msg.is_meta and msg.type in ABS0_GLOBAL_META:
                    insert_after = i + 1
                else:
                    insert_idx = i
                    break
            else:
                insert_idx = None

            if insert_idx is None:
                track.insert(insert_after, MetaMessage('marker', text='pre-roll', time=delay))
            else:
                track[insert_idx].time += delay

        mid.save(out_path)
        return

    # --- APPEND ---
    delay = bars * _ticks_per_bar(mid, use_last_ts=append_use_last_ts, default_ts=default_ts)
    delay = max(1, delay)  # at least 1 tick to avoid zero-length inserts
    last_tempo = _last_tempo_mpqn(mid)
    cidx = _find_conductor_track_idx(mid)
    ctrack = mid.tracks[cidx]

    # Extend the conductor track with a duplicate tempo meta before EOT
    eot_idx = None
    for i in range(len(ctrack) - 1, -1, -1):
        if ctrack[i].is_meta and ctrack[i].type == 'end_of_track':
            eot_idx = i
            break

    if eot_idx is None:
        # No EOT: simply append tempo-at-delay then an EOT(0)
        ctrack.append(MetaMessage('set_tempo', tempo=last_tempo, time=delay))
        ctrack.append(MetaMessage('end_of_track', time=0))
    else:
        eot = ctrack.pop(eot_idx)
        eot_delta = eot.time
        ctrack.insert(eot_idx, MetaMessage('set_tempo', tempo=last_tempo, time=eot_delta + delay))
        ctrack.insert(eot_idx + 1, MetaMessage('end_of_track', time=0))

    # Extra safety: add a dedicated padding track with a neutral channel event at the same time
    pad = MidiTrack()
    pad.append(MetaMessage('track_name', name='post-roll', time=0))
    pad.append(Message('control_change', channel=0, control=64, value=0, time=delay))
    pad.append(MetaMessage('end_of_track', time=0))
    mid.tracks.append(pad)

    mid.save(out_path)




def convert_lyrics(lyric_seq, ly2idx):
    pinyin_to_chars = {}
    for char in ly2idx:
        py = lazy_pinyin(char)[0]
        if py not in pinyin_to_chars:
            pinyin_to_chars[py] = []
        pinyin_to_chars[py].append(char)
    
    all_lyric_words = []
    for lyric_list in lyric_seq:
        seq_lyric_words = []
        for ly in lyric_list:
            if ly == "#":  
                # just skip this symbol
                continue
            if ly in ly2idx:
                seq_lyric_words.append(ly2idx[ly])
            else:
                py = lazy_pinyin(ly)[0]
                if py in pinyin_to_chars and pinyin_to_chars[py]:
                    replacement = pinyin_to_chars[py][0]
                    seq_lyric_words.append(ly2idx[replacement])
                else:
                    raise ValueError(f"字符 '{ly}' 沒有在 ly2idx 中找到同音字")
        all_lyric_words.append(seq_lyric_words)
    return all_lyric_words

def pad_sequence(seq, maxlen, pad_value):
  assert pad_value is not None
  seq.extend( [pad_value for _ in range(maxlen- len(seq))] )
  return seq

def get_encoder_input_data(seq_lyrics):
  enc_padding_mask = np.ones((len(seq_lyrics), config['data']['enc_seqlen']), dtype=bool)
  enc_padding_mask[:, :2] = False
  padded_enc_input = np.full((len(seq_lyrics), config['data']['enc_seqlen']), dtype=int, fill_value=len(lyric2idx)+1)
  enc_lens = np.zeros((len(seq_lyrics),))
  ind=0
  for lis in seq_lyrics:
    lis.insert(0,len(lyric2idx))
    enc_lens[ind]=len(lis)
    enc_padding_mask[ind, :len(lis)] = False
    within_seq_events = pad_sequence(lis, config['data']['enc_seqlen'], len(lyric2idx)+1)
    within_seq_events = np.array(within_seq_events)
    padded_enc_input[ind, :] = within_seq_events[:config['data']['enc_seqlen']]
    ind=ind+1
  return padded_enc_input, enc_padding_mask, enc_lens

###########################################
# little helpers
###########################################
def word2event(word_seq, idx2event):
  return [ idx2event[w] for w in word_seq ]

def get_beat_idx(event):
  return int(event.split('_')[-1])


###########################################
# sampling utilities
###########################################
def temperatured_softmax(logits, temperature):
  try:
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    assert np.count_nonzero(np.isnan(probs)) == 0
  except:
    print ('overflow detected, use 128-bit')
    logits = logits.astype(np.float128)
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    probs = probs.astype(float)
  return probs

def nucleus(probs, p):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][1]
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:3] # just assign a value
    candi_probs = np.array([probs[i] for i in candi_index], dtype=np.float64)
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word

########################################
# generation
########################################
def get_semantic_embedding(model, enc_input,enc_padding_mask):
  # reshape
  batch_inp = enc_input.permute(1, 0).long().to(device)
  batch_padding_mask = enc_padding_mask.bool().to(device)

  # get latent conditioning vectors
  with torch.no_grad():
    piece_latents = model.get_semantic_emb(
      batch_inp, padding_mask=batch_padding_mask)
  return piece_latents

def generate_on_latent_ctrl_vanilla_truncate(model, latents, seq_lyric_list,ND, Align, PM, DM, PV, DV, PR, DR, MCD, DMM, AA, CM, struct, key, emotion, pos, tone,event2idx, idx2event, max_events=12800, primer=None,nucleus_p=0.9, temperature=1.2):
  latent_placeholder = torch.zeros(max_events, 1, latents.size(-1)).to(device)
  ND_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  Align_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  PM_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  DM_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  PV_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  DV_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  PR_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  DR_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  MCD_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  DMM_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  AA_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  CM_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  struct_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  key_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  emotion_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  pos_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  tone_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  if primer is None:
    generated = [event2idx['SEQ_None']]
  else:
    generated = [event2idx[e] for e in primer]
    latent_placeholder[:len(generated), 0, :] = latents[0].squeeze(0)
    if config['model']['f_pos'] and config['model']['f_tone']:
      pos_placeholder[:len(generated), 0] = pos[0]
      tone_placeholder[:len(generated), 0] = tone[0]
    if config['model']['use_musc_ctls']:
      ND_placeholder[:len(generated), 0] = ND[0]
      Align_placeholder[:len(generated), 0] = Align[0]
      PM_placeholder[:len(generated), 0] = PM[0]
      DM_placeholder[:len(generated), 0] = DM[0]
      PV_placeholder[:len(generated), 0] = PV[0]
      DV_placeholder[:len(generated), 0] = DV[0]
      PR_placeholder[:len(generated), 0] = PR[0]
      DR_placeholder[:len(generated), 0] = DR[0]
      MCD_placeholder[:len(generated), 0] = MCD[0]
      DMM_placeholder[:len(generated), 0] = DMM[0]
      AA_placeholder[:len(generated), 0] = AA[0]
      CM_placeholder[:len(generated), 0] = CM[0]
      struct_placeholder[:len(generated), 0] = struct[0]
      key_placeholder[:len(generated), 0] = key
      emotion_placeholder[:len(generated), 0] = emotion

  target_bars, generated_bars = latents.size(0), 0
  steps = 0
  time_st = time.time()
  last_start_tick = 0
  cur_dur=0
  cur_bar=0
  failed_cnt = 0
  failed_eos=0
  seq_num_align=0

  cur_input_len = len(generated)
  generated_final = deepcopy(generated)
  entropies = []

  while generated_bars < target_bars:
    if len(generated) == 1:
      dec_input = numpy_to_tensor([generated], device=device).long()
    else:
      dec_input = numpy_to_tensor([generated], device=device).permute(1, 0).long()

    latent_placeholder[len(generated)-1, 0, :] = latents[ generated_bars ]
    if config['model']['f_pos'] and config['model']['f_tone']:
      pos_placeholder[len(generated)-1, 0] = pos[ generated_bars ]
      tone_placeholder[len(generated)-1, 0] = tone[ generated_bars ]
    if config['model']['use_musc_ctls']:
      ND_placeholder[len(generated)-1, 0] = ND[ generated_bars ]
      Align_placeholder[len(generated)-1, 0] = Align[ generated_bars ]
      PM_placeholder[len(generated)-1, 0] = PM[ generated_bars ]
      DM_placeholder[len(generated)-1, 0] = DM[ generated_bars ]
      PV_placeholder[len(generated)-1, 0] = PV[ generated_bars ]
      DV_placeholder[len(generated)-1, 0] = DV[ generated_bars ]
      PR_placeholder[len(generated)-1, 0] = PR[ generated_bars ]
      DR_placeholder[len(generated)-1, 0] = DR[ generated_bars ]
      MCD_placeholder[len(generated)-1, 0] = MCD[ generated_bars ]
      DMM_placeholder[len(generated)-1, 0] = DMM[ generated_bars ]
      AA_placeholder[len(generated)-1, 0] = AA[ generated_bars ]
      CM_placeholder[len(generated)-1, 0] = CM[ generated_bars ]
      struct_placeholder[len(generated)-1, 0] = struct[ generated_bars ]
      key_placeholder[len(generated)-1, 0] = key
      emotion_placeholder[len(generated)-1, 0] = emotion
 
    dec_seg_emb = latent_placeholder[:len(generated), :]
    dec_pos=pos_placeholder[:len(generated), :]
    dec_tone=tone_placeholder[:len(generated), :]
    dec_ND = ND_placeholder[:len(generated), :]
    dec_Align = Align_placeholder[:len(generated), :]
    dec_PM = PM_placeholder[:len(generated), :]
    dec_DM = DM_placeholder[:len(generated), :]
    dec_PV = PV_placeholder[:len(generated), :]
    dec_DV = DV_placeholder[:len(generated), :]
    dec_PR = PR_placeholder[:len(generated), :]
    dec_DR = DR_placeholder[:len(generated), :]
    dec_MCD = MCD_placeholder[:len(generated), :]
    dec_DMM = DMM_placeholder[:len(generated), :]
    dec_AA = AA_placeholder[:len(generated), :]
    dec_CM = CM_placeholder[:len(generated), :]
    dec_struct = struct_placeholder[:len(generated), :]
    dec_key = key_placeholder[:len(generated), :]
    dec_emotion = emotion_placeholder[:len(generated), :]
    # sampling
    with torch.no_grad():
      logits = model.generate(dec_input, dec_seg_emb, dec_struct, dec_key, dec_emotion, dec_ND, dec_Align, dec_PM, dec_PV, dec_PR, dec_DMM, dec_AA, dec_CM, dec_DM, dec_DV, dec_DR, dec_MCD, None, dec_pos,dec_tone)
    logits = tensor_to_numpy(logits[0])
    probs = temperatured_softmax(logits, temperature)
    word = nucleus(probs, nucleus_p)
    if len(seq_lyric_list[generated_bars])==seq_num_align:
      if word!=event2idx['SEQ_None']:
        word=event2idx['SEQ_None']
    else:
      if word==event2idx['SEQ_None']:
        logits[-2]=min(logits)
        probs = temperatured_softmax(logits, temperature)
        word = nucleus(probs, nucleus_p)

    word_event = idx2event[word]
    # print("generated word_event****:",word_event)

    if 'Beat' in word_event:
      event_pos = get_beat_idx(word_event)
      start_tick = cur_bar * DEFAULT_BAR_RESOL + event_pos * (DEFAULT_BAR_RESOL // DEFAULT_FRACTION)
      if not start_tick >= (last_start_tick+cur_dur):
        failed_cnt += 1
        print ('[info] position not increasing, failed cnt:', failed_cnt)
        if failed_cnt >= 128:
          print ('[FATAL] model stuck, exiting ...')
          return generated, False
        continue
      else:
        last_start_tick=start_tick
        failed_cnt = 0

    if 'Bar' in word_event and len(generated)>2:
      cur_bar+=1
    if 'SEQ' in word_event:
      generated_bars += 1
      seq_num_align=0
    if 'ALIGN' in word_event:
      seq_num_align=seq_num_align+1
    if 'Note_Duration' in word_event:
      cur_dur=int(word_event.split('_')[-1])
    if generated_bars < target_bars - 1 and word_event == 'EOS_None':
      failed_eos += 1
      print ('[info] error EOS occurs, failed eos:', failed_eos)
      if failed_eos >= 128:
        print ('[FATAL] model stuck, eos error ...')
        return generated, False
      continue

    if len(generated) > max_events or (word_event == 'EOS_None' and generated_bars == target_bars - 1):
      generated_bars += 1
      generated.append(event2idx['SEQ_None'])
      print ('[info] gotten eos')
      break

    generated.append(word)
    generated_final.append(word)
    entropies.append(entropy(probs))

    cur_input_len += 1
    steps += 1

    assert cur_input_len == len(generated)

  assert generated_bars == target_bars
  print ('-- generated events:', len(generated_final))
  print ('-- time elapsed: {:.2f} secs'.format(time.time() - time_st))
  return generated_final[:-1], True


if __name__ == "__main__":
  import opencc
  from find_similar_lyrics_pickles import find_best_matches
  # Define the segment label dictionary
  segment_dict = {'<Verse>': 0, '<Chorus>': 1, '<Insertion>': 2, '<Bridge>': 3, '<Outro>': 4}

  # Initialize the OpenCC converter for traditional to simplified Chinese
  converter = opencc.OpenCC('t2s')

  # Read the lyrics.txt file
  with open(lyrics_path, 'r', encoding='utf-8') as f:
      original_text = f.read()

  # Convert traditional Chinese to simplified Chinese
  simplified_text = converter.convert(original_text)

  # Split the text into "lines" based on whitespace, excluding segment markers
  lines = [line.replace('#', '') for line in simplified_text.split() if line not in segment_dict]
  seq_lyrics = [list(line) for line in lines]

  # Process the text line-by-line to assign segment labels
  segment_labels = []
  current_segment = None  # Track the current segment index
  for line in simplified_text.split():
      line = line.strip()  # Remove leading/trailing whitespace
      if not line:  # Skip empty lines
          continue
      if line in segment_dict:  # Check if the line is a segment marker
          current_segment = segment_dict[line]
      else:
          # This is a sentence; assign the current segment's index
          if current_segment is not None:
              segment_labels.append(current_segment)
          else:
              # Handle case where a sentence appears before any segment marker
              raise ValueError(f"Sentence '{line}' found before any segment marker")
  lyrics_length = len(seq_lyrics)
  # Output results
  print("lines = " + str(lines))
  print("seq_lyrics = " + str(seq_lyrics))
  print("segment_labels = " + str(segment_labels))
  print("lyrics_length = " + str(lyrics_length))
  print("lyrics_path", lyrics_path)
  print("seq_lyrics = " , seq_lyrics)
  merged_lyrics = list(chain.from_iterable(seq_lyrics))
  seq_lyrics_tokens=convert_lyrics(seq_lyrics, lyric2idx)
  enc_inp, enc_padding_mask, enc_lens = get_encoder_input_data(seq_lyrics_tokens)
  folder = "./sentence_struct_1000"
  matches = find_best_matches(folder, segment_labels, lines)
  piece = matches.split('/')[-1]
  # piece = '伴自己-常石磊-90-B.pkl'
  print("piece", piece)
  BPM = piece.split('-')[-2]

  pos_all=[]
  tone_all=[]
  for seq_lyric in seq_lyrics:
    str_lyric = "".join(seq_lyric)
    results=pypinyin.pinyin(str_lyric, style=Style.TONE3, heteronym=False)
    tone_seq=[]
    for ii in range(len(seq_lyric)):
      tone_value=results[ii][0][-1]
      if tone_value.isdigit():
        tone_seq.append(int(tone_value))
      else:
        tone_seq.append(5)
    tone_all.append(tone_seq)

    words=pseg.cut(str_lyric) 
    pos_seq=[]
    for word, flag in words:
      for jj in range(len(word)):
        pos_seq.append(flag)
    pos_all.append(pos_seq)

  mconf = config['model']
  if mconf['f_pos'] and mconf['f_tone']:
    pos2idx={'a': 0, 'ad': 1, 'ag': 2, 'an': 3, 'b': 4, 'c': 5, 'd': 6, 'df': 7, 'dg': 8, 'e': 9, 'eng': 10, 'f': 11, 'g': 12, 'h': 13, 'i': 14, 'j': 15, 'k': 16, 'l': 17, 'm': 18, 'mq': 19, 'n': 20, 'ng': 21, 'nr': 22, 'nrfg': 23, 'nrt': 24, 'ns': 25, 'nt': 26, 'nz': 27, 'o': 28, 'p': 29, 'q': 30, 'r': 31, 'rr': 32, 'rz': 33, 's': 34, 't': 35, 'tg': 36, 'u': 37, 'ud': 38, 'ug': 39, 'uj': 40, 'ul': 41, 'uv': 42, 'uz': 43, 'v': 44, 'vd': 45, 'vg': 46, 'vi': 47, 'vn': 48, 'vq': 49, 'x': 50, 'y': 51, 'yg': 52, 'z': 53, 'zg': 54}
    tone2idx={1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    pos_tokens=[]
    tone_tokens=[]
    seqs=len(pos_all)
    for s in range(seqs):
      num_lyrics=len(pos_all[s])
      for n in range(num_lyrics):
        pos_tokens.append(pos2idx[pos_all[s][n]])
        tone_tokens.append(tone2idx[tone_all[s][n]])
  else:
    pos_tokens=None
    tone_tokens=None

  model = CSLL2M(
    mconf['enc_n_layer'], mconf['enc_n_head'], mconf['enc_d_model'], mconf['enc_d_ff'],
    mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'], mconf['dec_d_ff'],
    mconf['d_latent'], mconf['d_embed'], len(event2idx)+1,len(lyric2idx)+2,len(event2idx),
    d_pos_emb=mconf['d_pos_emb'], d_tone_emb=mconf['d_tone_emb'],
    d_struct_emb=mconf['d_struct_emb'], d_key_emb=mconf['d_key_emb'], d_emotion_emb=mconf['d_emotion_emb'], 
    d_PM_emb=mconf['d_PM_emb'], d_PV_emb=mconf['d_PV_emb'], d_PR_emb=mconf['d_PR_emb'], d_DMM_emb=mconf['d_DMM_emb'], d_AA_emb=mconf['d_AA_emb'], d_CM_emb=mconf['d_CM_emb'], 
    d_DM_emb=mconf['d_DM_emb'], d_DV_emb=mconf['d_DV_emb'], d_DR_emb=mconf['d_DR_emb'], d_MCD_emb=mconf['d_MCD_emb'],
    d_ND_emb=mconf['d_ND_emb'], d_Align_emb=mconf['d_Align_emb'],
    d_learned_features=mconf['d_learned_features'],
    n_pos_cls=mconf['n_pos_cls'], n_tone_cls=mconf['n_tone_cls'],
    n_struct_cls=mconf['n_struct_cls'], n_key_cls=mconf['n_key_cls'], n_emotion_cls=mconf['n_emotion_cls'], 
    n_PM_cls=mconf['n_PM_cls'], n_PV_cls=mconf['n_PV_cls'], n_PR_cls=mconf['n_PR_cls'], n_DMM_cls=mconf['n_DMM_cls'], n_AA_cls=mconf['n_AA_cls'], n_CM_cls=mconf['n_CM_cls'], 
    n_DM_cls=mconf['n_DM_cls'], n_DV_cls=mconf['n_DV_cls'], n_DR_cls=mconf['n_DR_cls'], n_MCD_cls=mconf['n_MCD_cls'],
    n_ND_cls=mconf['n_ND_cls'], n_Align_cls=mconf['n_Align_cls'],
    f_pos=mconf['f_pos'], f_tone=mconf['f_tone'],
    f_struct=mconf['f_struct'], f_key=mconf['f_key'], f_emotion=mconf['f_emotion'], 
    f_PM=mconf['f_PM'], f_PV=mconf['f_PV'], f_PR=mconf['f_PR'], f_DMM=mconf['f_DMM'], f_AA=mconf['f_AA'], f_CM=mconf['f_CM'], 
    f_DM=mconf['f_DM'], f_DV=mconf['f_DV'], f_DR=mconf['f_DR'], f_MCD=mconf['f_MCD'],
    f_ND=mconf['f_ND'], f_Align=mconf['f_Align'],
    f_leared_features=mconf['f_leared_features'],
    use_musc_ctls=mconf['use_musc_ctls']).to(device)
  model.eval()
  pretrained_dict=torch.load(ckpt_path, map_location='cpu')

  adjusted_weights = {}
  for k, v in pretrained_dict.items():
    if k in model.state_dict():
      adjusted_weights[k] = v 

  try:
    model.load_state_dict(pretrained_dict, strict=False)
  except:
    model.load_state_dict(adjusted_weights, strict=False)

  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

    
  enc_inp = numpy_to_tensor(enc_inp, device=device)
  enc_padding_mask = numpy_to_tensor(enc_padding_mask, device=device)
       
  p_latents= get_semantic_embedding(model, enc_inp,enc_padding_mask)


  
  if config['model']['use_musc_ctls']:
    p_ND = pickle_load(os.path.join('./share_StatisticalAttributes/ND_seq_d64', piece))
    p_Align = pickle_load(os.path.join('./share_StatisticalAttributes/Align_seq_d64', piece))
    p_PM = pickle_load(os.path.join('./share_StatisticalAttributes/PM_seq_d64', piece))
    p_MCD = pickle_load(os.path.join('./share_StatisticalAttributes/MCD_seq_d64', piece))
    p_DMM = pickle_load(os.path.join('./share_StatisticalAttributes/DMM_seq_d64', piece))
    p_AA = pickle_load(os.path.join('./share_StatisticalAttributes/AA_seq_d64', piece))
    p_CM = pickle_load(os.path.join('./share_StatisticalAttributes/CM_seq_d64', piece))
    p_PV = pickle_load(os.path.join('./share_StatisticalAttributes/PV_seq_d64', piece))
    p_PR = pickle_load(os.path.join('./share_StatisticalAttributes/PR_seq_d64', piece))
    p_DM = pickle_load(os.path.join('./share_StatisticalAttributes/DM_seq_d64', piece))
    p_DV = pickle_load(os.path.join('./share_StatisticalAttributes/DV_seq_d64', piece))
    p_DR = pickle_load(os.path.join('./share_StatisticalAttributes/DR_seq_d64', piece))

    p_struct =torch.tensor(segment_labels, device=device) # {'Verse':0, 'Chorus':1, 'Insertion':2, 'Bridge':3, 'Outro':4}
    p_key =torch.tensor([7], device=device) #{'A':0, 'Ab':1, 'Am':2, 'B':3, 'Bb':4, 'Bbm':5, 'Bm':6, 'C':7, 'C#m':8, 'Cm':9, 'D':10, 'D#m':11, 'Db':12, 'Dm':13, 'E':14, 'Eb':15, 'Em':16, 'F':17, 'F#':18, 'F#m':19, 'Fm':20, 'G':21, 'G#m':22, 'Gm':23}
    p_emotion =torch.tensor([2], device=device)  #{'Neutral':0, 'Negative':1, 'Positive':2}
  else:
    p_ND = None
    p_Align =None
    p_PM = None
    p_DM = None
    p_PV=None
    p_DV =None
    p_PR =None
    p_DR =None
    p_MCD =None
    p_DMM =None
    p_AA =None
    p_CM =None
    print("using structure, key, emotion conditions")
    p_struct =torch.tensor(segment_labels, device=device) # {'Verse':0, 'Chorus':1, 'Insertion':2, 'Bridge':3, 'Outro':4}
    p_key =torch.tensor([7], device=device) #{'A':0, 'Ab':1, 'Am':2, 'B':3, 'Bb':4, 'Bbm':5, 'Bm':6, 'C':7, 'C#m':8, 'Cm':9, 'D':10, 'D#m':11, 'Db':12, 'Dm':13, 'E':14, 'Eb':15, 'Em':16, 'F':17, 'F#':18, 'F#m':19, 'Fm':20, 'G':21, 'G#m':22, 'Gm':23}
    p_emotion =torch.tensor([0], device=device)  #{'Neutral':0, 'Negative':1, 'Positive':2}



  max_retries = 5  # 每個 sample 最多重試次數
  generated = 0    # 已成功產生的件數（只在成功時 +1）

  while generated < n_samples_per_piece:
      # 只有成功後才遞增，所以檔名會是連號、不會被失敗嘗試佔走
      out_file = os.path.join(out_dir, 'sample01')

      success = False
      for attempt in range(1, max_retries + 1):
          song, success = generate_on_latent_ctrl_vanilla_truncate(
              model, p_latents, seq_lyrics, p_ND, p_Align, p_PM, p_DM, p_PV, p_DV,
              p_PR, p_DR, p_MCD, p_DMM, p_AA, p_CM, p_struct, p_key, p_emotion,
              pos_tokens, tone_tokens, event2idx, idx2event,
              nucleus_p=config['generate']['nucleus_p'],
              temperature=config['generate']['temperature']
          )
          if success:
              break
          print(f"[retry] sample {generated+1} attempt {attempt}/{max_retries} failed; retrying...")

      if not success:
          print(f"[skip] sample {generated+1} failed after {max_retries} attempts.")
          # 這裡可以選擇直接 continue（跳過這個編號），或是不要+1、繼續嘗試；我這裡選擇跳過。
          # 若你想一定要湊滿件數，把上面的 continue 改成 `continue` 前先不做任何事即可（本例已是如此）。
          continue

      # 成功才往下做
      song = word2event(song, idx2event)
      _ = REMIaligned2midi(merged_lyrics, song, 90, out_file + '.mid')

      add_silent_bars(out_file + '.mid', out_file + '.mid', bars=4, where="prepend")
      a = MidiFile(out_file + '.mid')
      print("After prepend (s):", a.length)

      add_silent_bars(out_file + '.mid', out_file + '.mid', bars=4, where="append")
      meta_path = out_file+'.json'
      meta = {
          "bpm": BPM,
      }
      import json
      with open(meta_path, "w", encoding="utf-8") as f:
          json.dump(meta, f, ensure_ascii=False, indent=2)
      a = MidiFile(out_file + '.mid')
      print("After append (s):", a.length)

      generated += 1

