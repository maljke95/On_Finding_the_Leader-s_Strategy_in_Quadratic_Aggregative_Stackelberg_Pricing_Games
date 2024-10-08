# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 17:22:50 2021

@author: marko
"""
import numpy as np

def pinvs(A):
    
    A_inv = np.zeros((len(A), len(A)))
    for i in range(len(A)):
        
        if not(A[i,i] == 0.0):
            
            A_inv[i,i] = 1.0/A[i,i]
            
    return A_inv
    
    