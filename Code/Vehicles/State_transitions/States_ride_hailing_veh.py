# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 16:43:37 2022

@author: marko
"""

#----- List of transition functions -----

def f_idle(action):
    
    if action == 0:        
        return "idle"
    
    elif action == 1:
        return "pick_passenger"
    
    elif action == 100:
        return "park"
    
def f_idle_electric(action):
    
    if action == 0:        
        return "idle"
    
    elif action == 1:
        return "pick_passenger"
    
    elif action == 2:
        return "charge"
    
    elif action == 100:
        return "park"
    
def f_pick_passenger(action):
    
    if action == 0:
        return "pick_passenger"
    
    elif action == 1:
        return "transporting_passenger"
    
def f_transporting_passenger(action):
    
    if action == 0:
        return "transporting_passenger"
    
    elif action == 1:
        return "idle"
    
def f_park(action):
    
    return "park"

def f_charge(action):
    
    if action == 0:
        return "charge"
    
    elif action == 1:    
        return "park"
        