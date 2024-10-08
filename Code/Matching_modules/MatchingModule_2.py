# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 21:28:43 2022

@author: marko
"""
import numpy as np
#import cvxpy as cp
import gurobipy as gp

from Matching_modules.MatchingModule_template import MatchingAlgorithm

class MatchingAlgorithm_2(MatchingAlgorithm):
    
    def __init__(self, max_cost=1000000):
        
        super(MatchingAlgorithm_2, self).__init__(max_cost)
                            
    # def match(self, N_veh, N_pass):

    #     print("veh, pass", N_veh, N_pass)
    #     dim = N_veh*N_pass
        
    #     cost_vec = self.cost_matrix.reshape(dim)
        
    #     x = cp.Variable(dim, integer=True)
        
    #     obj = cp.Minimize(cost_vec @ x)
        
    #     for i in range(N_pass):
            
    #         for j in range(N_pass):
                
    #             if i == j:
    #                 current = np.array(N_veh*[1.0]).reshape(1, -1)
    #             else:
    #                 current = np.array(N_veh*[0.0]).reshape(1, -1)
                    
    #             if j == 0:
    #                 curr_row = current
    #             else:
    #                 curr_row = np.concatenate((curr_row, current), axis=1)
            
    #         if i == 0:   
    #             C1_l = curr_row
    #             C2_l = np.eye(N_veh)                
    #         else:    
    #             C1_l = np.concatenate((C1_l, curr_row), axis=0)
    #             C2_l = np.concatenate((C2_l, np.eye(N_veh)), axis=1)
                
    #     c1_r = np.array(N_pass*[1.0])
    #     c2_r = np.copy(self.vehicle_capacities)
        
    #     print("c1r, c2r", len(c1_r), len(c2_r))
    #     print("C1_l,", C1_l.shape[0], C1_l.shape[1])
    #     #print("C2_l,", C2_l.shape[0], C2_l.shape[1])
        
    #     constraints = [C1_l @ x == c1_r, C2_l @ x <= c2_r]
        
    #     prob = cp.Problem(obj, constraints)
    #     prob.solve()
        
    #     print(x.value)
    #     self.matching_vector = np.array(x.value)
        
    def match(self, N_veh, N_pass):
        
        dim = N_veh*N_pass
        
        MIP = gp.Model()
        MIP.Params.OutputFlag = 0
        
        cost_vec = self.cost_matrix.reshape(dim)
        
        x = MIP.addMVar(dim, vtype = gp.GRB.BINARY, name='x')
        MIP.update()
        
        MIP.setObjective(cost_vec @ x, gp.GRB.MINIMIZE)
        MIP.update()
        
        for i in range(N_pass):
            
            for j in range(N_pass):
                
                if i == j:
                    current = np.array(N_veh*[1.0]).reshape(1, -1)
                else:
                    current = np.array(N_veh*[0.0]).reshape(1, -1)
                    
                if j == 0:
                    curr_row = current
                else:
                    curr_row = np.concatenate((curr_row, current), axis=1)
            
            if i == 0:   
                C1_l = curr_row
                C2_l = np.eye(N_veh)                
            else:    
                C1_l = np.concatenate((C1_l, curr_row), axis=0)
                C2_l = np.concatenate((C2_l, np.eye(N_veh)), axis=1)
                
        c1_r = np.array(N_pass*[1.0])
        c2_r = np.copy(self.vehicle_capacities)
               
        MIP.addConstr(C1_l @ x == c1_r)
        MIP.addConstr(C2_l @ x <= c2_r)
        MIP.update()
        
        MIP.optimize()
        
        #print('Objective function value: %.2f' % MIP.objVal)
        
        self.assignment_matrix = np.array(MIP.getAttr('X')).reshape((N_pass, -1))   
                    
        
                    
                
                
                
        
        
        
        
        
        
        