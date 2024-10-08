# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 15:47:55 2022

@author: marko
"""

#----- List of transition functions -----

def f_transporting_passenger(action):
    
    if action == 0:        
        return "transporting_passenger"
    
    elif action == 1:        
        return "park"
    
def f_park(action):
    
    return "park"