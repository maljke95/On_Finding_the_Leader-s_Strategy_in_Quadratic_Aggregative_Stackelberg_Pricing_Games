# -*- coding: utf-8 -*-
from quadprog import solve_qp
import numpy as np

class Projected_Krasnoselskij_iteration():
    
    def __init__(self, gamma, alpha=0.5):
        
        self.gamma = gamma
        self.alpha = alpha
        
    def update(self, xi, grad, K_l, k_r):
        
        x_hat = xi - self.gamma*grad

        qp_G = np.eye(len(xi))
        qp_a = x_hat
        qp_C = np.transpose(-K_l)
        qp_b = np.squeeze(-k_r)
        
        sol,_,_,_,_,_ = solve_qp(qp_G, qp_a, qp_C, qp_b) 

        return 0.5*xi + 0.5*np.squeeze(np.array(sol))        
        
