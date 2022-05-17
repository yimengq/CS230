# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
#1. Réfléchir à la structure du programme (boucles à utiliser)
#2. Quelles variables utiliser?

A = np.random.randn(4,3)
B = np.sum(A, axis = 1, keepdims = True)
print(B.shape)