#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 18:20:17 2016

@author: marshi
"""

import numpy as np
import pretty_midi

def make_song(file,instr=0):
    res = 960
    tempo = 120
    pm = pretty_midi.PrettyMIDI(resolution=res, initial_tempo=tempo) #pretty_midiオブジェクトを作ります
    instrument = pretty_midi.Instrument(instr) #instrumentはトラックに相当します。
    
    note = pretty_midi.Note(velocity=100, pitch=60, start=0, end=0.25)
    instrument.notes.append(note)
    note = pretty_midi.Note(velocity=100, pitch=62, start=0.25, end=0.5)
    instrument.notes.append(note)
    note = pretty_midi.Note(velocity=100, pitch=64, start=0.5, end=0.75)
    instrument.notes.append(note)
    note = pretty_midi.Note(velocity=100, pitch=65, start=0.75, end=1)
    instrument.notes.append(note)
    note = pretty_midi.Note(velocity=100, pitch=67, start=1, end=1.5)
    instrument.notes.append(note)
    note = pretty_midi.Note(velocity=100, pitch=69, start=1.5, end=1.75)
    instrument.notes.append(note)
    note = pretty_midi.Note(velocity=100, pitch=65, start=1.75, end=2)
    instrument.notes.append(note)
    note = pretty_midi.Note(velocity=100, pitch=64, start=2, end=2.25)
    instrument.notes.append(note)
    note = pretty_midi.Note(velocity=100, pitch=62, start=2.5, end=2.75)
    instrument.notes.append(note)
    note = pretty_midi.Note(velocity=100, pitch=60, start=3, end=4)
    instrument.notes.append(note)
    
    pm.instruments.append(instrument)
    pm.write(file) #midiファイルを書き込みます。

for i in range(128):
    make_song('mid/%d.mid'%i,i)