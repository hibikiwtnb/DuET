"""Glissandi named by the caller, written as one wavy line from the first note to the last.

The model gives every input note its own score position, which turns a glissando into a crowded run of 32nds. It has
no way to say "glissando", so the caller says which notes are one (Config.glissandi: performance seconds and pitches,
for example from Piano MIDI Cleanup's *.gliss.json). This does not detect anything. Other notes change only when struck
during the line: they go back to where they were played and become the second voice of the staff.
"""
from fractions import Fraction as F
import math

from .spelling import STEP_PC

MATCH_SECONDS = 0.005
POSITION_GRID = F(1, 4)     # the ends of the line go on sixteenths: its inner rhythm is not notated
END_NOTE = F(1, 4)


def snap(value):
    """The nearest sixteenth, half up so that a tie does not depend on the beat."""
    return F(math.floor(value / float(POSITION_GRID) + 0.5)) * POSITION_GRID


def natural_spelling(pitch):
    """A white key written as itself: in E major a white-key glissando has F natural, not E sharp."""
    return next(((step, 0) for step, pc in STEP_PC.items() if pc == pitch % 12), None)


def apply(notes, runs, ctx):
    """Keep the first and last note of each run with the line between them and drop the rest. Sets ctx.gliss_spans."""
    ctx.gliss_spans = []
    last_end = None
    for number, run in enumerate(runs, 1):
        wanted = run['notes']
        found = []
        for seconds, pitch in wanted:
            match = [n for n in notes if n['pitch'] == pitch and abs(n['t'] - seconds) <= MATCH_SECONDS]
            if match:
                found.append(match[0])
        if len(found) != len(wanted) or len(found) < 2:
            ctx.log(f"WARNING gliss: run {number} matched {len(found)} of {len(wanted)} notes; left as it was")
            continue
        found.sort(key=lambda n: n['t'])
        first, last = found[0], found[-1]
        # Performed positions: the model stretches a glissando onto its 32nd grid, so its own positions overrun.
        start, end = snap(first['x']), snap(last['x'])
        if last_end is not None and start <= last_end:
            start = last_end + POSITION_GRID
        if end <= start:
            end = start + POSITION_GRID
        last_end = end
        staff = {'right': 0, 'left': 1}.get(run.get('hand'), first['staff'])
        for note, position, duration, kind in ((first, start, end - start, 'start'), (last, end, END_NOTE, 'stop')):
            note.update(on=position, dur=duration, staff=staff, grace=False, bass_grace=False, roll=False,
                        gliss=kind, gliss_staff=staff, gliss_scale=run.get('scale'))
        inner = {id(n) for n in found[1:-1]}
        notes[:] = [n for n in notes if id(n) not in inner]
        ctx.gliss_spans.append((staff, start, end))
        ctx.log(f"gliss: run {number} {first['pitch']} to {last['pitch']}, {len(found)} notes written as a line "
                f"({ctx.where(start)})")
    replace_crossing_notes(notes, ctx)
    return notes


def replace_crossing_notes(notes, ctx):
    """Notes struck during a line go back where they were played.

    The model's stretched glissando carried them along, by up to half a beat in Dream Believers m136.
    """
    moved = 0
    for _, start, end in ctx.gliss_spans:
        for note in notes:
            if 'gliss' in note or note['grace'] or not float(start - POSITION_GRID) <= note['x'] < float(end):
                continue
            position = snap(note['x'])
            if abs(float(note['on']) - note['x']) > float(POSITION_GRID) and note['on'] != position:
                note['on'] = position
                moved += 1
    if moved:
        ctx.log(f"gliss: {moved} crossing notes moved to where they were played")


def finish(notes):
    """After the passes that change staves and spelling: a line stays on one staff, and white keys stay natural."""
    for note in notes:
        if 'gliss' in note:
            note['staff'] = note['gliss_staff']
            if note['gliss_scale'] == 'white':
                note['spell'] = natural_spelling(note['pitch'])


def split_layers(pool, staff, spans):
    """The line is the main voice of its staff; other notes struck during it are the second voice."""
    for span_staff, start, end in spans:
        if span_staff != staff:
            continue
        for note in pool:
            if 'gliss' in note:
                note['layer'] = 'A'
            elif start <= note['on'] < end:
                note['layer'] = 'B'


def restore_lengths(pool, spans):
    """Duration cleanup must not end the first note before the line reaches the last one."""
    for note in pool:
        if note.get('gliss') == 'start':
            for _, start, end in spans:
                if start == note['on']:
                    note['dur'] = end - start


def notation(note, is_attack, is_last_piece):
    """The <notations> content of one written piece of a note: the line leaves the last piece of its first note."""
    kind = note.get('gliss')
    if kind == 'stop' and is_attack:
        return '<glissando type="stop" line-type="wavy" number="1"/>'
    if kind == 'start' and is_last_piece:
        return '<glissando type="start" line-type="wavy" number="1">gliss.</glissando>'
    return ''
