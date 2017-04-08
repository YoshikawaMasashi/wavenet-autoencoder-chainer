#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 10:59:47 2017

@author: marshi
"""

import numpy as np
import scipy as sp
import scipy.io.wavfile
import chainer
from chainer import optimizers,serializers
import chainer.functions as F
import chainer.links as L
from model import WavenetAutoencoder

x = np.load("x.npy")
pitch = np.load("pitch.npy")
pitch = np.eye(7)[pitch].T
seqlen = pitch.shape[1]
seqlen -= seqlen%512
x,t = x[:,:seqlen],x[:,1:seqlen+1]
pitch = pitch[:,1:seqlen+1]
pitch = pitch.astype(np.float32)

model = WavenetAutoencoder()
opt = optimizers.Adam()
opt.setup(model)

#serializers.load_npz("model.model",model)
for epoch in range(100000):
    print(epoch)
    idx = np.arange(128)
    np.random.shuffle(idx)
    for i,idx_ in enumerate(idx):
        batch_t,batch_x = t[[idx_]],x[[idx_]]
        z = model.encoder(batch_t)
        t_ = model.decoder(batch_x, z, pitch)
        loss = F.softmax_cross_entropy(t_[:,:,:,0],batch_t)

        model.cleargrads()
        loss.grad = np.ones(loss.shape, dtype=np.float32)
        loss.backward()
        opt.update()

        if i % 16 == 0:
            t_label = np.argmax(F.log_softmax(t_[:,:,:,0]).data, axis=1)
            acc = np.mean(t_label == batch_t)
            err = np.abs(t_label - batch_t)
            err = np.mean(err)
            print('loss %.3f'%loss.data,'acc %.3f'%acc,'err %.3f'%err)
    #serializers.save_npz("model.model",model)
