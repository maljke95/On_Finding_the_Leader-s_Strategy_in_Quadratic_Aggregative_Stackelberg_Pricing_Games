# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 21:28:43 2022

@author: marko
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from Matching_modules.MatchingModule_template import MatchingAlgorithm

class MatchingAlgorithm_1(MatchingAlgorithm):
    
    def __init__(self, max_cost=1000000):
        
        super(MatchingAlgorithm_1, self).__init__(max_cost)
        
    def match(self, N_veh, N_pass):
        
        row_ind, col_ind = linear_sum_assignment(self.cost_matrix)
        
        self.assignment_matrix = np.zeros((N_pass, N_veh))
        
        for idx in range(len(row_ind)):
            
            row = row_ind[idx]
            col = col_ind[idx]
            self.assignment_matrix[row, col] = 1
               
        
        
                    
        
                    
                
                
                