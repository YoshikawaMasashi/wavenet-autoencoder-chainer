#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 12:34:46 2017

@author: marshi
"""

from model import WavenetAutoencoder
import numpy as np
import chainer.functions as F
from chainer import optimizers,serializers
import matplotlib.pyplot as plt
import scipy as sp
import scipy.io.wavfile

def inv_mu_law(data, mu=255):
    data -= data.min()
    data /= data.max()
    data = data*2 -1
    data = np.sign(data)*(1/mu)*(np.power(1+mu,np.abs(data))-1)
    return data
    
x = np.load("x.npy")
pitch = np.load("pitch.npy")
pitch = np.eye(7)[pitch].T
seqlen = pitch.shape[1]
seqlen -= seqlen%512
x,t = x[:,:seqlen],x[:,1:seqlen+1]
pitch = pitch[:,1:seqlen+1]
pitch = pitch.astype(np.float32)

model = WavenetAutoencoder()
serializers.load_npz("model.model",model)

generate_idx = 108
batch_t,batch_x = t[[generate_idx]],x[[generate_idx]]
z = model.encoder(batch_t, test=True)
t_ = model.decode_generator(z, pitch, test=True)

wav = np.array(t_, dtype=np.float64)
wav = inv_mu_law(wav)
sp.io.wavfile.write("generated%d.wav"%generate_idx, 4400,wav)