# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 18:06:29 2022

@author: marko
"""
import numpy as np
import itertools
import matplotlib.pyplot as plt

#import gurobipy as gp

from quadprog import solve_qp

class Company(object):
    
    def __init__(self, Map, idx=None):
        
        self.company_ID = idx
        
        self.Map = Map
        
        self.Ni = int(0)
        
        self.Vehicles = []
        
        self.p_occ = None
        self.avg_earning = None
        
        self.average_cost_of_arrival = None
        self.average_charging_demand = None
                
        #----- Parameters of the cost function -----
        
        self.cost_function = None
        self.iterative_alg = None
        
        #----- Parameters of the feasibility constraint -----
        
        self.Fi = np.array(self.Map.M*[0.0])
        
        self.K_l = None
        self.k_r = None
        
        #----- Projected gradient descent -----
        
        self.xi = None
        self.grad = None
        self.sigma_x = None
        self.loss_buffer = []
        self.decision_buffer = []        
        
    def reset_buffers(self):
        
        self.xi = None
        self.grad = None
        self.sigma_x = None
        self.loss_buffer = []
        self.decision_buffer = []
        
    def convert_to_vehicles_from_sim(self, RHVs):
        
        for veh in RHVs:
                      
            if veh.companyID == self.company_ID and veh.current_state == "charge":
                
                veh.update_vehicle_for_charging(self.Map.stations)
                self.Vehicles.append(veh)
                
        self.Ni = len(self.Vehicles)
                
    def generate_matrices_for_loss(self, N_max, N_des, Q):
        
        Ni = self.Ni
        Ai = 2*Ni**2*Q
        Bi = Ni*Q
        ci = -Ni*Q @ N_max
        Di = Ni*self.average_charging_demand
        fi = Ni*(self.avg_cost_of_arrival - self.Map.ExpProfit)
  
        return Ai, Bi, ci, Di, fi, Ni
            
    def set_cost_and_iterative_alg(self, cost_function, iterative_alg):
        
        self.cost_function = cost_function
        self.iterative_alg = iterative_alg
        
    def generate_feasibility_set(self):
        
        M = self.Map.M
        
        for veh in self.Vehicles:
            
            self.Fi[veh.feasible_stations] += 1
            
        #----- Create the feasibility constraint -----
        
        list_of_index = list(np.arange(M))
        all_combinations = []
        
        for r in range(len(list_of_index) + 1):

            combinations_object = itertools.combinations(list_of_index, r)
            combinations_list = list(combinations_object)
            all_combinations += combinations_list

        all_combinations = all_combinations[1:]        
        
        L = np.zeros((2**M-1, M))
        R = np.zeros(2**M-1)     

        for comb_id in range(len(all_combinations)):

            comb = list(all_combinations[comb_id])
            L[comb_id, comb] = 1 
            
            for veh in self.Vehicles:
                
                for station_id in veh.feasible_stations:
                    
                    if station_id in comb:
                        
                        R[comb_id] += 1
                        break 
                    
            R[comb_id] = R[comb_id] - 1.0*len(comb)
            if R[comb_id]<0.0:
                R[comb_id] = 0.0
                
        L = np.copy(L[:-1, :])
        
        R = np.copy(R[:-1])
        R = R/(1.0*self.Ni)
        
        self.K_l = np.concatenate((-np.eye(M), np.ones((1, M)), -np.ones((1, M)), np.copy(L)), axis=0)        
        self.k_r = np.concatenate((np.zeros((M, 1)), np.ones((1,1)), -np.ones((1,1)), np.copy(R).reshape((len(R), 1))), axis=0)        
        
        x0 = np.array(M*[1/M])
        
        self.xi = self.compute_feasible_initialization(x0)
        
        
    def compute_feasible_initialization(self, x0):
        
        qp_G = np.eye(self.Map.M)
        qp_a = x0
        qp_C = np.transpose(-self.K_l)
        qp_b = np.squeeze(-self.k_r)

        sol,_,_,_,_,_ = solve_qp(qp_G, qp_a, qp_C, qp_b) 

        x0_feas = np.squeeze(np.array(sol))
        
        return x0_feas
        
    def prepare_cumulative_state(self, p_occ, avg_earning):
        
        M = self.Map.M
        
        self.p_occ = p_occ
        self.avg_earning = avg_earning
        
        avg_distance = np.zeros(M)
        avg_charge = np.zeros(M)
        
        for veh in self.Vehicles:
            
            ch_feas = veh.desired_battery_level*np.ones(M) - (veh.current_battery_level*np.ones(M) - 100.0/veh.max_range*veh.dist)  
            ch_feas = np.maximum(ch_feas, np.array(M*[0.0]))
            avg_charge += ch_feas
            
            avg_distance += np.array(veh.dist)
        
        for i in range(M):
            
            if self.Fi[i] > 0.0:
                
                avg_distance[i] /= self.Fi[i]
                avg_charge[i]   /= self.Fi[i]
                
            else:
                
                avg_distance[i] = 0.0
                avg_charge[i]   = 0.0
        
        self.avg_cost_of_arrival = avg_earning * np.diag(p_occ) @ avg_distance        
        self.average_charging_demand = np.diag(avg_charge)
        
    def random_initialization(self):
        
        x = np.random.rand(self.Map.M)
        x = x/np.sum(x)
        
        self.xi = self.compute_feasible_initialization(x)
        
    def collect_sigma_x(self, sigma_x):
        
        self.sigma_x = sigma_x
        
    def broadcast_decision(self):

        return self.Ni*self.xi
                   
    def update_decision(self):
        
        self.grad = self.cost_function.calc_grad(self.xi, self.sigma_x)
        self.loss_buffer.append(self.cost_function.evaluate_loss(self.xi, self.sigma_x))
        self.decision_buffer.append(self.xi)
        
        self.xi = self.iterative_alg.update(self.xi, self.grad, self.K_l, self.k_r)

    def plot_decision_making(self):
        
        fig1, ax1 = plt.subplots()
        k = np.arange(len(self.decision_buffer))
        
        decision_matrix = np.array(self.decision_buffer)
        
        for i in range(self.Map.M):
            
            ax1.plot(k, np.squeeze(decision_matrix[:, i]), label='Station '+str(i))
        
        ax1.grid('on')
        
        ax1.legend(loc='lower right')
        ax1.set_xlabel("Iteration [k]")
        ax1.set_ylim(0, 0.8)
        ax1.set_ylabel(r'$x_{i}^{j}$')
        fig1.suptitle('Company '+str(self.company_id))
        #plt.show()
        plt.close()

    def check_optimality_of_company(self):
        
        print("Loss with NE: "+ str(self.cost_function.evaluate_loss(self.xi, self.sigma_x)))
        
        qp_G, qp_a, qp_C, qp_b = self.cost_function.matrices_for_company_optimality_check()
        
        sol,_,_,_,_,_ = solve_qp(qp_G, qp_a, qp_C, qp_b) 
        
        x_opt = np.squeeze(np.array(sol))
        print("Loss with calculated x_opt: " + str(self.cost_function.evaluate_loss(x_opt, self.sigma_x)))   

    # def assign_vehicles_to_charging_stations_single_surge_pricing(self, v_avr, Tv):
        
    #     M = self.Map.M

    #     N_des_com = self.broadcast_decision()
    #     N_des_com = np.round(N_des_com)
    #     N_des_com[-1] = self.Ni - np.sum(N_des_com[:-1])

    #     MIP = gp.Model()
    #     MIP.Params.OutputFlag = 0
    #     MIP.Params.TimeLimit = 60
        
    #     x  = MIP.addMVar(len(self.Vehicles)*M, vtype = gp.GRB.BINARY, name='x')   
    #     ro = MIP.addMVar(M, lb=0.0, name='fare') #lb is lower bound
        
    #     MIP.update()
        
    #     prices_NE = self.cost_function.calculate_prices(self.xi, self.sigma_x)
        
    #     Si_set = []
        
    #     for veh_id, veh in enumerate(self.Vehicles):
            
    #         Dv, neg_exp_revenue, Kl_a, kr_a, feasible_set = veh.prepare_assignment_procedure_for_vehicle(Tv, self.p_occ, self.avg_earning, self.Map.ExpProfit)
            
    #         Si = np.zeros((M, M*self.Ni))
            
    #         for i in range(M):
    #             Si[i, veh_id*M + i] = 1
                
    #         Si_set.append(Si)
                
    #         if veh_id == 0:
    #             A = np.eye(M)
    #         else:
    #             A = np.concatenate((A, np.eye(M)), axis=1)
            
    #         Pi = np.ones(M) @ Si
            
    #         MIP.addConstr(Pi @ x == 1)
            
    #         if not ((Kl_a is None) or (kr_a is None)):
    #             MIP.addConstr(Kl_a @ Si @ x == kr_a)
            
    #         MIP.update()
            
    #         for x_f in feasible_set:
                
    #             Ki = Dv @ prices_NE + neg_exp_revenue
    #             Li = np.diag(np.squeeze(self.p_occ)) @ np.diag(M*[Tv*v_avr])
                
    #             Mi = Li @ x_f
                
    #             Ni = prices_NE @ Dv @ x_f + x_f @ neg_exp_revenue
                
    #             Yi = - Li @ Si
                
    #             MIP.addConstr(Ki @ Si @ x + ro @ Yi @ x + ro @ Mi - Ni <= 0)
    #             MIP.update()
                
    #     M1 = 0.5* A.T @ A
    #     m1 = N_des_com @ A
        
    #     MIP.setObjective(x @ M1 @ x - m1 @ x, gp.GRB.MINIMIZE)
    #     MIP.params.NonConvex = 2
        
    #     MIP.update()
        
    #     MIP.optimize()
        
    #     print(30*'=')
    #     print('Objective function value: %.2f' % MIP.objVal)
        
    #     x_total = np.array(MIP.getAttr('X'))
    #     x_final = np.copy(x_total[:len(self.Vehicles)*M])
    #     e_final = np.copy(x_total[len(self.Vehicles)*M:])
        
    #     suma = np.zeros(self.Map.M)
        
    #     for veh_id in range(len(self.Vehicles)):
            
    #         veh = self.Vehicles[veh_id]            
    #         suma = suma + Si_set[veh_id] @ x_final           
            
    #     print('Sum: ', suma) 
    #     print('N_des: ', N_des_com)
    #     print('E_final: ', np.sum(e_final))    

    #     return x_final, e_final, suma, N_des_com         

    # def assign_vehicles_to_charging_stations_individual_surge_pricing(self, v_avr, Tv):                
                
    #     M = self.Map.M

    #     N_des_com = self.broadcast_decision()
    #     N_des_com = np.round(N_des_com)
    #     N_des_com[-1] = self.Ni - np.sum(N_des_com[:-1])

    #     MIP = gp.Model()
    #     MIP.Params.OutputFlag = 0
    #     MIP.Params.TimeLimit = 60
        
    #     x  = MIP.addMVar(len(self.Vehicles)*M, vtype = gp.GRB.BINARY, name='x')
    #     ro = MIP.addMVar(len(self.Vehicles)*M, lb=0.0, name='fare')                                  #lb is lower bound
        
    #     MIP.update()
        
    #     prices_NE = self.cost_function.calculate_prices(self.xi, self.sigma_x)
        
    #     Si_set = []
        
    #     for veh_id, veh in enumerate(self.Vehicles):
            
    #         Dv, neg_exp_revenue, Kl_a, kr_a, feasible_set = veh.prepare_assignment_procedure_for_vehicle(Tv, self.p_occ, self.avg_earning, self.Map.ExpProfit)
            
    #         Si = np.zeros((M, M*self.Ni))
            
    #         for i in range(M):
    #             Si[i, veh_id*M + i] = 1
                
    #         Si_set.append(Si)
                
    #         if veh_id == 0:
    #             A = np.eye(M)
    #         else:
    #             A = np.concatenate((A, np.eye(M)), axis=1)
            
    #         Pi = np.ones(M) @ Si
            
    #         MIP.addConstr(Pi @ x == 1)
            
    #         if not ((Kl_a is None) or (kr_a is None)):
    #             MIP.addConstr(Kl_a @ Si @ x == kr_a)
            
    #         MIP.update()
            
    #         for x_f in feasible_set:
                
    #             Ki = Dv @ prices_NE + neg_exp_revenue
    #             Li = np.diag(np.squeeze(self.p_occ)) @ np.diag(M*[Tv*v_avr])
                
    #             Mi = Si.T @ Li @ x_f
                
    #             Ni = prices_NE @ Dv @ x_f + x_f @ neg_exp_revenue
                
    #             Yi = - Si.T @ Li @ Si
                
    #             MIP.addConstr(Ki @ Si @ x + ro @ Yi @ x + ro @ Mi - Ni <= 0)
    #             MIP.update()
                
    #     M1 = 0.5* A.T @ A
    #     m1 = N_des_com @ A
        
    #     MIP.setObjective(x @ M1 @ x - m1 @ x, gp.GRB.MINIMIZE)
    #     MIP.params.NonConvex = 2
        
    #     MIP.update()
        
    #     MIP.optimize()
        
    #     print(30*'=')
    #     print('Objective function value: %.2f' % MIP.objVal)
        
    #     x_total = np.array(MIP.getAttr('X'))
    #     x_final = np.copy(x_total[:len(self.Vehicles)*M])
    #     e_final = np.copy(x_total[len(self.Vehicles)*M:])
        
    #     suma = np.zeros(self.Map.M)
    #     list_of_surges = []
        
    #     for veh_id in range(len(self.Vehicles)):
            
    #         veh = self.Vehicles[veh_id]            
    #         suma = suma + Si_set[veh_id] @ x_final 
    #         surge = Si_set[veh_id] @ e_final
    #         list_of_surges.append(surge)
        
    #     list_of_surges = np.array(list_of_surges)
    #     list_of_surges = list_of_surges.T
        
    #     print('Sum: ', suma) 
    #     print('N_des: ', N_des_com)
    #     print('E_final: ', np.sum(e_final))    
    #     print('Surges: ', list_of_surges[:, :20])  
        
    #     return x_final, e_final, suma, N_des_com, list_of_surges

                     
            
                
