"""Bar lines from the input MIDI's own beat grid.

A performance MIDI that was already put on a beat grid (a constant meter, notes
near the grid's eighths) states where every bar starts. The model still predicts
bars itself and, in syncopated passages, can drift a beat or more off the grid,
which shifts every later bar line. When the input is on a grid, each generated
note that the model placed a whole number of eighths away from its grid position
is moved back by those eighths; finer rhythm (16ths, tuplets) stays the model's.
"""
from collections import Counter

import mido
import numpy as np

BEAT = 480                 # codec ticks per quarter
ALIGNED_MEDIAN = 0.15      # eighths; median onset distance from the eighth grid (random onsets: 0.25)
STEP = BEAT // 2           # a drift moves notes by whole eighths
SHIFT_TOL = 0.15           # beats; a displacement this close to whole eighths is a drift


def load(path, onsets):
    """Grid of a constant-meter MIDI, or None (with the reason) when it has none.
    onsets: note onsets in seconds, as read_performance_midi returns them."""
    mid = mido.MidiFile(str(path))
    tempos, meters = [], set()
    for track in mid.tracks:
        tick = 0
        for message in track:
            tick += message.time
            if message.type == 'set_tempo':
                tempos.append((tick, message.tempo))
            elif message.type == 'time_signature':
                meters.add((message.numerator, message.denominator))
    if len(meters) > 1:
        return None, 'meter changes'
    num, den = meters.pop() if meters else (4, 4)
    tempos = sorted(dict(sorted(tempos)).items()) or [(0, 500000)]
    if tempos[0][0] > 0:
        tempos.insert(0, (0, 500000))
    # seconds at each tempo change, then a piecewise-linear seconds -> beats map
    ticks = np.array([t for t, _ in tempos], float)
    rates = np.array([v for _, v in tempos], float) / 1e6 / mid.ticks_per_beat   # seconds per tick
    secs = np.concatenate(([0.0], np.cumsum(np.diff(ticks) * rates[:-1])))
    on = np.asarray(onsets, float)
    seg = np.searchsorted(secs, on, side='right') - 1
    beats = (ticks[seg] + (on - secs[seg]) / rates[seg]) / mid.ticks_per_beat   # quarter notes, like BEAT
    eighths = beats * 2
    median = float(np.median(np.abs(eighths - np.round(eighths))))
    if median > ALIGNED_MEDIAN:
        return None, f'not on a beat grid (median {median:.2f} eighth off)'
    return dict(beats=beats, bar=round(BEAT * 4 * num / den), num=num, den=den, offset=None,
                median=median, moved=0), None


def correct(grid, codec, owned, rows, anchor):
    """Move each owned note's predicted row back onto the grid. Returns the new rows."""
    bar = grid['bar']
    rows = rows.copy()
    pos = (rows[:, 0] - codec.bar_offset + anchor) * bar + rows[:, 1] * BEAT + codec.fractions[rows[:, 2]]
    want = grid['beats'][owned] * BEAT
    if grid['offset'] is None:
        # keep the model's bar numbering: the whole-bar difference it settles on first
        grid['offset'] = Counter(np.round((want - pos) / bar).astype(int)).most_common(1)[0][0] * bar
    delta = want - grid['offset'] - pos
    shift = np.round(delta / STEP) * STEP
    new = (pos + shift).astype(np.int64)
    fraction = {int(t): i for i, t in enumerate(codec.fractions)}
    move = ((shift != 0) & (np.abs(delta - shift) <= SHIFT_TOL * BEAT) & (new >= 0)
            & np.array([int(t) % BEAT in fraction for t in new], bool))
    length = int(np.flatnonzero(codec.tick_values['bar_length'] == bar)[0])
    for k in np.flatnonzero(move):
        b, inside = divmod(int(new[k]), bar)
        rows[k, 0] = b - anchor + codec.bar_offset
        rows[k, 1] = inside // BEAT
        rows[k, 2] = fraction[inside % BEAT]
    rows[:, 5] = length
    rows[:, 6] = grid['num'] - 1
    rows[:, 7] = codec.denominators.index(grid['den'])
    grid['moved'] += int(move.sum())
    return rows
