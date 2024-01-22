# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 18:23:30 2019

@author: pheno

plot loss from checkpoints
"""

import torch

import matplotlib.pyplot as plt
import numpy as np


fname = './data/small_training_set/checkpoints_hybridnet_stoc_pg/checkpoint_02000.tar'

cp = torch.load(fname, map_location=torch.device('cpu'))
loss = np.array(cp['loss'])

plt.figure(figsize=(25,5))
#plt.loglog(loss[-6000:])
plt.plot(loss[-500:])
#plt.ylim(0, 180)
plt.xlabel('no. of training steps')
#plt.xlim(1000,10000)
plt.ylabel('total loss')
#plt.title('Offset = 0.1')
plt.savefig('./data/pictures/loss_02000.png')