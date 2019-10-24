#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:21:43 2019

@author: nozim

"""

from lr1 import LR1, model

LR1 = LR1()

trainApple = LR1.loadTrainApple()
trainPapaya = LR1.loadTrainPapaya()

model = model(LR1)
model.train(trainApple, 'apple')
model.train(trainPapaya, 'papaya')

model.predict(LR1.loadTestApple(), "apple")
model.predict(LR1.loadTestPapaya(), "papaya")
