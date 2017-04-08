#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 18:20:17 2016

@author: marshi
"""

import scipy as sp
import scipy.io.wavfile
import numpy as np

def mu_law(data, mu=255):
    data -= data.min()
    data /= data.max()
    data = data*2 -1
    data = np.sign(data)*np.log(1+mu*np.abs(data))/np.log(1+mu)
    data = np.round((data + 1) / 2 * mu).astype(np.int32)
    return data
    
x = []
pitch = [1,2,3,4,5,5,6,4,3,0,2,0,1,1,1,1]
for i in range(128):
    wav = 'wav/'+str(i)+'.wav'
    rate,data = sp.io.wavfile.read(wav)
    x.append(mu_law(data).reshape((1,)+data.shape))
x = np.concatenate(x)
pitch = np.repeat(pitch,rate)
pitch = pitch[::4]

np.save("x.npy",x)
np.save("pitch.npy",pitch)