#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 18:20:17 2016

@author: marshi
"""

import glob
from midi2audio import FluidSynth
import scipy as sp
import scipy.io.wavfile
import numpy as np

fs = FluidSynth("TimGM6mb.sf2")
for i in range(128):
    mid = 'mid/'+str(i)+'.mid'
    wav = 'wav/'+str(i)+'.wav'
    fs.midi_to_audio(mid, wav)
    rate,data = sp.io.wavfile.read(wav)
    
    #ぴったし4秒にする処理
    data = data[80:]
    data = data[:-(data.shape[0]-rate*4)]
    data = data.mean(axis=1)
    data /= np.abs(data).max()
    data = data[::10]
    sp.io.wavfile.write(wav,int(rate/10),data)