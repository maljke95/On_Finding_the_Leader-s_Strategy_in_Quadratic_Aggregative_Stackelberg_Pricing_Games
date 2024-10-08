# -*- coding: utf-8 -*-

class Government():
    
    def __init__(self):
        
        self.Ag = None
        self.bg = None
        
    def generate_government(self, Ag, bg):
        
        self.Ag = Ag
        self.bg = bg
        
    def calculate_welfare(self, sigma_x):

        return 0.5*sigma_x @ self.Ag @ sigma_x + self.bg @ sigma_x

