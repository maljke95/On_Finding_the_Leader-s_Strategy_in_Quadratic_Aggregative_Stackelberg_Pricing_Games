# -*- coding: utf-8 -*-

from Charging_balancing_modules.utils_s import pinvs
import numpy as np

#----- Loss under perfect information -----

class Loss_under_perfect_info():
    
    def __init__(self):
        
        self.Ai = None
        self.Bi = None
        self.ci = None
        self.Di = None
        self.fi = None
        
        self.Ni = None
        
        self.Ag = None
        self.bg = None
        
        #----- Derived parameters -----
        
        self.Ai_bar = None
        self.Bi_bar = None
        self.delta_i = None
        
        #----- Robustness terms -----
        
        self.robust_coeff = None
        self.delta_Di = None
        self.delta_inv_Di = None
        self.Gi = None
        
    def define_loss(self, Ai, Bi, ci, Di, fi, Ni, Ag, bg, robust_coeff=0.0, scale_param=1.0):
        
        self.Ai = Ai
        self.Bi = Bi
        self.ci = ci
        self.Di = Di
        self.fi = fi
        
        self.Ni = Ni
        
        self.Ag = Ag
        self.bg = bg 
        
        self.Ai_bar = self.Ni**2 * self.Ag - self.Ai

        self.Bi_bar = self.Ni * self.Ag - self.Bi

        self.delta_i = self.Ni * self.bg - self.ci - self.fi
        
        #----- Calculate robustness parameters -----
        
        self.robust_coeff = robust_coeff
        
        demands = np.diag(self.Di)
        
        non_zero_demands = demands[demands > 0.0]
        
        min_demand = np.min(non_zero_demands)

        var = (robust_coeff * min_demand/scale_param)**2
        mean = np.array(len(demands)*[0.0])
        cov = var*np.eye(len(non_zero_demands))

        non_zero_delta_Di = np.random.multivariate_normal(mean, cov)
        
        delta_Di = np.zeros((len(demands), len(demands)))
        
        idx = 0
        for i in range(len(demands)):           
            if demands[i]>0.0:
                delta_Di[i, i] = non_zero_delta_Di[idx]
                idx += 1
                
        self.delta_Di = delta_Di
        
        perturbed_Di = self.Di + delta_Di
        inv_perturbed_Di = pinvs(perturbed_Di)
        inv_Di = pinvs(self.Di)
        
        self.delta_inv_Di = inv_perturbed_Di - inv_Di
        self.Gi = self.Di @ self.delta_inv_Di
               
    def calc_grad(self, xi, sigma_x):
        
        grad = self.Ni**2 * self.Ag @ xi + self.Ni*(self.Ag @ (sigma_x - self.Ni*xi) + self.bg) + self.Gi @ (self.Ai_bar @ xi + self.Bi_bar @ (sigma_x - self.Ni*xi) + self.delta_i)
        return grad
        
    def evaluate_loss(self, xi, sigma_x):
        
        loss = 0.5*self.Ni**2* xi @ self.Ag @ xi + xi @ (self.Ni*(self.Ag @ (sigma_x - self.Ni*xi) + self.bg)) + xi @ self.Gi @ (0.5*self.Ai_bar @ xi + self.Bi_bar @ (sigma_x - self.Ni*xi) + self.delta_i)
        return loss
    
    def calculate_prices(self, xi, sigma_x):
        
        prices = (pinvs(self.Di) + self.delta_inv_Di) @ (0.5 * self.Ai_bar @ xi + self.Bi_bar @ (sigma_x - self.Ni*xi) + self.delta_i)
        
        return prices
    
    def  matrices_for_company_optimality_check(self):
        
        pass
    
#----- Loss with predefined prices -----

class Loss_for_fixed_prices():
    
    def __init__(self):
        
        self.Ai = None
        self.Bi = None
        self.ci = None
        self.Di = None
        self.fi = None
        
        self.Ni = None
        
        self.Ag = None
        self.bg = None
        
        self.p = None
        
    def define_loss(self, Ai, Bi, ci, Di, fi, Ni, Ag, bg, robust_coeff=0.0, scale_param=1.0):
        
        self.Ai = Ai
        self.Bi = Bi
        self.ci = ci
        self.Di = Di
        self.fi = fi
        
        self.Ni = Ni
        
        self.Ag = Ag
        self.bg = bg 

        #----- Calculate robustness parameters -----
        
        self.robust_coeff = robust_coeff
        
        demands = np.diag(self.Di)
        
        non_zero_demands = demands[demands > 0.0]
        
        min_demand = np.min(non_zero_demands)

        var = (robust_coeff * min_demand/scale_param)**2
        mean = np.array(len(demands)*[0.0])
        cov = var*np.eye(len(non_zero_demands))

        non_zero_delta_Di = np.random.multivariate_normal(mean, cov)
        
        delta_Di = np.zeros((len(demands), len(demands)))
        
        idx = 0
        for i in range(len(demands)):           
            if demands[i]>0.0:
                delta_Di[i, i] = non_zero_delta_Di[idx]
                idx += 1
        
        self.delta_Di = delta_Di
        
        perturbed_Di = self.Di + delta_Di
        
        self.Di = perturbed_Di
        
    def set_fixed_prices(self, p):
        
        self.p = p
        
    def calc_grad(self, xi, sigma_x):
        
        grad = self.Ai @ xi + self.Bi @ (sigma_x - self.Ni*xi) + self.ci + self.fi + self.Di @ self.p
        return grad
        
    def evaluate_loss(self, xi, sigma_x):
        
        loss = 0.5 * xi @ self.Ai @ xi + xi @ self.Bi @ (sigma_x - self.Ni*xi) + xi @ (self.ci + self.fi + self.Di @ self.p)
        return loss
    
    def calculate_prices(self, xi, sigma_x):
        
        return self.p
    
    def  matrices_for_company_optimality_check(self):
        
        pass
    