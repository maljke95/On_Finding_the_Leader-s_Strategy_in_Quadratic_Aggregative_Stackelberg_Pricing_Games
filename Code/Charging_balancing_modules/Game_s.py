# -*- coding: utf-8 -*-

import numpy as np
import os

from Charging_balancing_modules.utils_s import *
from Charging_balancing_modules.Government_s import *
from Charging_balancing_modules.Company_s import *
from Charging_balancing_modules.Optimizer_s import *
from Charging_balancing_modules.GameCoord_s import *
from Charging_balancing_modules.CompanyLoss_s import *

#----- Loading the simulator -----

from Maps.MapShenzhen_electric import MapShenzhen_electric
from Vehicles.RideHailingVehicle_electric import RideHailingVehicle_electric
from Vehicles.State_transitions.States_ride_hailing_veh import f_idle_electric, f_pick_passenger, f_transporting_passenger, f_park, f_charge

class Game():
    
    def __init__(self):
        
        self.list_of_GameCoord = []
        
    def reset_game(self):
        
        self.list_of_GameCoord = []
    
    def define_parameters(self, sim, fixed_params=None):
        
        Ag=None 
        bg=None 
        N_max=None 
        N_des=None 
        Q=None 
        p_occ_belief=None 
        avg_earning_belief=None 
        gammas=None  
                
        #----- Read all values that are passed -----
        
        if not(fixed_params is None):
            
            Q     = fixed_params['Q']
            Ag    = fixed_params['Ag']
            N_des = fixed_params['N_des']
            N_max = fixed_params['N_max']
            p_occ_belief = fixed_params['p_occ_belief']
            avg_earning_belief = fixed_params['avg_earning_belief']
            gammas = fixed_params['gammas']
            bg = fixed_params['bg']
            
        #----- Set all values that are left as None -----
        
        if Q is None:
            
            Q = 0.1*np.diag([5.0, 1.0, 3.0, 2.0])
        
        if Ag is None:
                
            Ag = 2.5*Q

        if N_des is None:
            
            N_ref = sim.Map.N_des_coeff
            
            N_total = 0
            for veh in sim.RHVs:
                if veh.current_state == "charge":
                    N_total += 1
                
            N_des = np.array(len(N_ref)*[0.0]) 
                
            for i in range(len(N_ref)- 1):
                N_des[i] = np.floor(N_total/np.sum(N_ref) * N_ref[i])
                          
            N_des[-1] = N_total - np.sum(N_des)
              
        if N_max is None:
            
            N_max = np.array([10, 60, 40, 50])

        if p_occ_belief is None:

            p_occ_belief = 3*[np.array([0.35, 0.15, 0.20, 0.17])] 
            
        if avg_earning_belief is None:
            
            avg_earning_belief = np.array(3 * [1.0])
            
        if gammas is None:
            
            gammas = np.array(3 * [0.0000001*5])

        if bg is None:
            
            bg = -Ag @ N_des

        return  Ag, bg, N_max, N_des, Q, p_occ_belief, avg_earning_belief, gammas
    
    def generate_game_from_sim(self, sim, Ag, bg, N_max, N_des, Q, p_occ_belief, avg_earning_belief, gammas, list_of_games, fixed_params=None, robust_coeff=0.0, scale_param=1.0):
            
        m = sim.Map
        
        fixed_profit = fixed_params['fixed_exp_profit']
        
        if not fixed_profit is None:
            m.ExpProfit = fixed_profit 
            
        N_companies = sim.N_companies
        RHVs = sim.RHVs
        fixed_price = fixed_params['fixed_prices']
        
        self.generate_game_coordinators(list_of_games, m, N_des, N_max, Q, Ag, bg, N_companies, RHVs, p_occ_belief, avg_earning_belief, robust_coeff, scale_param, gammas, fixed_price)

    def Run_game_from_sim(self, Ag, bg, N_max, N_des, Q, p_occ_belief, avg_earning_belief, gammas, sim, fixed_params, plot_graphs=True, list_of_games=None, list_of_games_to_play=None, folder_name=None, robust_coeff=0.0, scale_param=1.0):

        if list_of_games is None:
            
            list_of_games = [0,1]
            
        #----- Generate parameters -----

        self.generate_game_from_sim(sim, Ag, bg, N_max, N_des, Q, p_occ_belief, avg_earning_belief, gammas, list_of_games, fixed_params, robust_coeff, scale_param)
                
        government_performance_logger = self.play_different_games(list_of_games_to_play, plot_graphs, folder_name)
            
        return government_performance_logger
        
    def generate_game_from_stored_data(self, Map, RHVs, N_companies, Ag, bg, N_max, N_des, Q, p_occ_belief, avg_earning_belief, gammas, list_of_games, fixed_price, robust_coeff=0.0, scale_param=1.0):
        
        m = Map
        
        self.generate_game_coordinators(list_of_games, m, N_des, N_max, Q, Ag, bg, N_companies, RHVs, p_occ_belief, avg_earning_belief, robust_coeff, scale_param, gammas, fixed_price)

    def Run_game_from_stored_data(self, Ag, bg, p_occ_belief, avg_earning_belief, Q, N_max, N_des, Map, RHVs, gammas, N_companies, fixed_price, plot_graphs=True, list_of_games=None, list_of_games_to_play=None, folder_name=None, robust_coeff=0.0, scale_param=1.0):
        
        if list_of_games is None:
            
            list_of_games = [0,1]
            
        #----- Generate parameters -----
                    
        self.generate_game_from_stored_data(Map, RHVs, N_companies, Ag, bg, N_max, N_des, Q, p_occ_belief, avg_earning_belief, gammas, list_of_games, fixed_price, robust_coeff, scale_param)

        government_performance_logger = self.play_different_games(list_of_games_to_play, plot_graphs, folder_name)
            
        return government_performance_logger

    def generate_game_coordinators(self, list_of_games, m, N_des, N_max, Q, Ag, bg, N_companies, RHVs, p_occ_belief, avg_earning_belief, robust_coeff, scale_param, gammas, fixed_price):
        
        for game_id in list_of_games:
            
            #----- Inverse Stackelberg game -----
            
            if game_id == 0:
                
                gc0 = GameCoordinator(Map=m, idx=game_id, N_des=N_des)
                G0 = Government()
                G0.generate_government(np.copy(Ag), np.copy(bg))
                gc0.set_gov(G0)
                
                for i in range(N_companies):
                    
                    c0 = Company(m, i)
                    c0.convert_to_vehicles_from_sim(RHVs)
                    c0.generate_feasibility_set()
                    c0.prepare_cumulative_state(np.copy(p_occ_belief[i]), np.copy(avg_earning_belief[i]))
                    Ai0, Bi0, ci0, Di0, fi0, Ni0 = c0.generate_matrices_for_loss(N_max, N_des, Q)
                    
                    #--- Cost function for perfect info ---
                    
                    cost_f0 = Loss_under_perfect_info()
                    cost_f0.define_loss(Ai0, Bi0, ci0, Di0, fi0, Ni0, Ag, bg, robust_coeff, scale_param)
                    
                    #--- Set iterative procedure and cost function ---
                    
                    iterative0 = Projected_Krasnoselskij_iteration(gammas[i])
                    c0.set_cost_and_iterative_alg(cost_f0, iterative0) 
                    
                    gc0.add_comp(c0)
                    
                self.list_of_GameCoord.append(gc0)
            
            #----- Stackelberg game -----
            
            if game_id == 1:
                
                gc1 = GameCoordinator(Map=m, idx=game_id, N_des=N_des)
                G1 = Government()
                G1.generate_government(np.copy(Ag), np.copy(bg))
                gc1.set_gov(G1) 
                
                for i in range(N_companies):
                
                    c1 = Company(m, i)
                    c1.convert_to_vehicles_from_sim(RHVs)
                    c1.generate_feasibility_set()
                    c1.prepare_cumulative_state(np.copy(p_occ_belief[i]), np.copy(avg_earning_belief[i]))
                    Ai1, Bi1, ci1, Di1, fi1, Ni1 = c1.generate_matrices_for_loss(N_max, N_des, Q)
                    
                    #--- Cost for fixed prices ---
                    
                    cost_f1 = Loss_for_fixed_prices() 
                    cost_f1.define_loss(Ai1, Bi1, ci1, Di1, fi1, Ni1, Ag, bg, robust_coeff, scale_param)
                    
                    #--- Set the fixed price for the game ---
                    
                    cost_f1.set_fixed_prices(fixed_price)
                    
                    #--- Set iterative procedure and cost function ---
                                           
                    iterative1 = Projected_Krasnoselskij_iteration(gammas[i])
                    c1.set_cost_and_iterative_alg(cost_f1, iterative1)
                    
                    gc1.add_comp(c1)
                    
                self.list_of_GameCoord.append(gc1)        
                
    def play_different_games(self, list_of_games_to_play, plot_graphs, folder_name):

        #----- Play different games -----
        
        government_performance_logger = []
                
        for game_id in list_of_games_to_play:
            
            gc_id = list_of_games_to_play.index(game_id)
            
            self.list_of_GameCoord[gc_id].play()  
            
            if plot_graphs:
                
                self.list_of_GameCoord[gc_id].plot_exp_accumulation_evolution(folder_name=folder_name)        
                self.list_of_GameCoord[gc_id].plot_NE_pricing(folder_name=folder_name)        
                self.list_of_GameCoord[gc_id].plot_losses(folder_name=folder_name)
            
            government_performance_logger.append(self.list_of_GameCoord[gc_id].list_of_welfare[-1])
            print(30*'=')
                
        if plot_graphs:
            
            list_of_centroids = self.list_of_GameCoord[0].Map.stations
            self.list_of_GameCoord[0].Map.voronoi_regions(list_of_centroids, show_plot=plot_graphs, save_fig=True, folder_name=folder_name)
            
        return government_performance_logger
    
    def save_info_about_game(self, fixed_params, folder_name):
        
        list_of_veh_idx = []
        list_of_cID = []
        list_of_previous_node = []
        list_of_previous_node_pos_x = []
        list_of_previous_node_pos_y = []
        list_of_veh_states = []
        
        list_of_next_node = []
        list_of_remaining_distance_to_next_node = []
        
        list_of_traversed_distances = []
        
        list_of_current_battery_level = []
        list_of_max_range = []
        list_of_charge_thr = []
        list_of_desired_battery_level = []
        
        N_vehicles = 0
        
        N_companies = len(self.list_of_GameCoord[0].Companies)
        
        list_of_stations     = self.list_of_GameCoord[0].Map.stations
        list_of_ExpProfit    = self.list_of_GameCoord[0].Map.ExpProfit
        list_of_N_des_coeff  = self.list_of_GameCoord[0].Map.N_des_coeff
        list_of_N_max        = self.list_of_GameCoord[0].Map.N_max
        list_of_gammas       = len(self.list_of_GameCoord[0].Companies)*[self.list_of_GameCoord[0].Companies[0].iterative_alg.gamma]
        
        Ag = self.list_of_GameCoord[0].Government.Ag
        bg = self.list_of_GameCoord[0].Government.bg
        
        p_occ_belief = len(self.list_of_GameCoord[0].Companies)*[self.list_of_GameCoord[0].Companies[0].p_occ]
        avg_earning_belief = len(self.list_of_GameCoord[0].Companies)*[self.list_of_GameCoord[0].Companies[0].avg_earning]
        
        Q = fixed_params['Q']
        
        for ind, comp in enumerate(self.list_of_GameCoord[0].Companies):
            
            N_vehicles  += len(comp.Vehicles)
            
            for veh in comp.Vehicles:
                
                if veh.current_state == "idle":            
                    list_of_veh_states.append(0)
                elif veh.current_state == "pick_passenger":            
                    list_of_veh_states.append(1)            
                elif veh.current_state == "transporting_passenger":            
                    list_of_veh_states.append(2)        
                elif veh.current_state == "charge":            
                    list_of_veh_states.append(3)  
                elif veh.current_state == "park":
                    list_of_veh_states.append(4)
                
                list_of_veh_idx.append(veh.ID)
                
                list_of_cID.append(veh.companyID)
                
                list_of_previous_node.append(veh.previous_node)
                list_of_previous_node_pos_x.append(veh.previous_node_pos[0])
                list_of_previous_node_pos_y.append(veh.previous_node_pos[1])
                
                list_of_traversed_distances.append(veh.traversed_distance)
                
                if not (veh.next_node is None):
                    list_of_next_node.append(veh.next_node)
                    list_of_remaining_distance_to_next_node.append(veh.remaining_distance_to_next_node)
                else:
                    list_of_next_node.append(-100)
                    list_of_remaining_distance_to_next_node.append(-100)                   
                
                list_of_current_battery_level.append(veh.current_battery_level)
                list_of_max_range.append(veh.max_range)
                list_of_charge_thr.append(veh.charge_thr)
                list_of_desired_battery_level.append(veh.desired_battery_level)
        
        #----- Save numpy arrays -----
        
        if not fixed_params['fixed_prices'] is None:
            
            np.save(folder_name + '/fixed_prices.npy', np.array(fixed_params['fixed_prices']))
        
        np.save(folder_name + '/list_of_veh_states.npy', np.array(list_of_veh_states))
        np.save(folder_name + '/N_companies.npy', np.array([N_companies]))
        
        np.save(folder_name + '/stations.npy', np.squeeze(np.array(list_of_stations)))
        np.save(folder_name + '/ExpProfit.npy', np.squeeze(np.array(list_of_ExpProfit)))
        np.save(folder_name + '/N_des_coeff.npy', np.squeeze(np.array(list_of_N_des_coeff)))
        np.save(folder_name + '/N_max.npy', np.squeeze(np.array(list_of_N_max)))
        np.save(folder_name + '/Ag.npy', np.squeeze(np.array(Ag)))
        np.save(folder_name + '/bg.npy', bg)
        np.save(folder_name + '/p_occ_belief.npy', np.squeeze(np.array(p_occ_belief)))
        
        np.save(folder_name + '/avg_earning_belief.npy', np.squeeze(np.array(avg_earning_belief)))
        np.save(folder_name + '/Q.npy', Q) 
        np.save(folder_name + '/list_of_gammas.npy',  np.squeeze(np.array(list_of_gammas)))
        
        np.save(folder_name + '/list_of_veh_idx.npy', np.squeeze(np.array(list_of_veh_idx)))
        np.save(folder_name + '/list_of_cID.npy', np.squeeze(np.array(list_of_cID)))
        np.save(folder_name + '/list_of_previous_node.npy', np.squeeze(np.array(list_of_previous_node)))
        np.save(folder_name + '/list_of_previous_node_pos_x.npy', np.squeeze(np.array(list_of_previous_node_pos_x)))
        np.save(folder_name + '/list_of_previous_node_pos_y.npy', np.squeeze(np.array(list_of_previous_node_pos_y)))
        np.save(folder_name + '/list_of_traversed_distances.npy', np.squeeze(np.array(list_of_traversed_distances)))
        
        np.save(folder_name + '/list_of_next_node.npy', np.squeeze(np.array(list_of_next_node)))
        np.save(folder_name + '/list_of_remaining_distance_to_next_node.npy', np.squeeze(np.array(list_of_remaining_distance_to_next_node)))
        
        np.save(folder_name + '/list_of_current_battery_level.npy', np.squeeze(np.array(list_of_current_battery_level)))
        np.save(folder_name + '/list_of_max_range.npy', np.squeeze(np.array(list_of_max_range)))
        np.save(folder_name + '/list_of_charge_thr.npy', np.squeeze(np.array(list_of_charge_thr)))
        np.save(folder_name + '/list_of_desired_battery_level.npy', np.squeeze(np.array(list_of_desired_battery_level)))
        
    def load_info_about_game(self, folder_name):
        
        #----- Setup Map -----
        
        batch_id = 2 
        arrivals_id = 0
        list_of_stations = np.load(folder_name + '/stations.npy')
        list_of_capacities = np.load(folder_name + '/N_max.npy')
        list_of_ExpProfit = np.load(folder_name + '/ExpProfit.npy')
        list_of_N_des_coeff = np.load(folder_name + '/N_des_coeff.npy') 
        
        path = os.getcwd()
        
        path_data           = path + '/Maps/Data/input_data.mat'
        path_arrivals       = path + '/Maps/Data/arrival batchs/arrivals_40k80k.mat'
        path_coordinates    = path + '/Maps/Data/Grid/coordinates.mat'
        
        Map = MapShenzhen_electric(batch_id, arrivals_id, list_of_stations, list_of_capacities, path_data, path_arrivals, path_coordinates)
        Map.setup_demand_based_values()
        Map.ExpProfit = list_of_ExpProfit
        Map.N_des_coeff = list_of_N_des_coeff

        allcoordinates = Map.coordinates
        alldists = Map.alldists
        allpaths = Map.allpaths
        allintersect = Map.allintersect
        capacity = 1
        distance_epsilon = 0.000001
        
        #----- Generate RHVs -----
        
        N_companies = np.load(folder_name + '/N_companies.npy')
        N_companies = N_companies[0]
        
        list_of_ride_hailing_vehicle_states = ["idle", "pick_passenger", "transporting_passenger", "park", "charge"]
        list_of_ride_hailing_vehicle_transitions = [f_idle_electric, f_pick_passenger, f_transporting_passenger, f_park, f_charge]
        
        list_of_veh_idx = np.load(folder_name + '/list_of_veh_idx.npy')
        list_of_cID = np.load(folder_name + '/list_of_cID.npy')
        list_of_previous_node = np.load(folder_name + '/list_of_previous_node.npy')
        list_of_veh_states = np.load(folder_name + '/list_of_veh_states.npy')
        
        list_of_next_node = np.load(folder_name + '/list_of_next_node.npy')
        list_of_remaining_distance_to_next_node = np.load(folder_name + '/list_of_remaining_distance_to_next_node.npy')
        list_of_traversed_distances = np.load(folder_name + '/list_of_traversed_distances.npy')
        
        list_of_current_battery_level = np.load(folder_name + '/list_of_current_battery_level.npy')
        list_of_max_range = np.load(folder_name + '/list_of_max_range.npy')
        list_of_charge_thr = np.load(folder_name + '/list_of_charge_thr.npy')
        list_of_desired_battery_level = np.load(folder_name + '/list_of_desired_battery_level.npy') 

        N_veh = len(list_of_previous_node)
        RHVs = []
        
        for i in range(N_veh):
            
            idx = list_of_veh_idx[i]
            cID = list_of_cID[i]
            start_node = list_of_previous_node[i]
            charge_thr = list_of_charge_thr[i]
            max_range = list_of_max_range[i]
            current_battery_level = list_of_current_battery_level[i]
            desired_battery_level = list_of_desired_battery_level[i]
            
            veh_state = list_of_veh_states[i]
            
            if veh_state == 0:            
                current_state = "idle"
            elif veh_state == 1:            
                current_state = "pick_passenger"            
            elif veh_state == 2:            
                current_state = "transporting_passenger"      
            elif veh_state == 3:            
                current_state = "charge" 
            elif veh_state == 4:
                current_state = "park"
                
            initial_state = current_state
                    
            veh = RideHailingVehicle_electric(idx, start_node, allintersect, allcoordinates, cID, list_of_ride_hailing_vehicle_states, list_of_ride_hailing_vehicle_transitions, alldists, charge_thr, max_range, current_battery_level, desired_battery_level, allpaths, capacity, initial_state, distance_epsilon)
            
            if list_of_next_node[i] == -100:
                veh.next_node = None
                veh.remaining_distance_to_next_node = None
            else:
                veh.next_node = list_of_next_node[i]
                veh.remaining_distance_to_next_node = list_of_remaining_distance_to_next_node[i]
                
            veh.traversed_distance = list_of_traversed_distances[i]
            
            RHVs.append(veh)
        
        #----- Load Simulation Parameters -----
        
        Ag = np.load(folder_name + '/Ag.npy')
        bg = np.load(folder_name + '/bg.npy')
        p_occ_belief = np.load(folder_name + '/p_occ_belief.npy')
        avg_earning_belief = np.load(folder_name + '/avg_earning_belief.npy')
        Q = np.load(folder_name + '/Q.npy')         
        
        N_max = list_of_capacities
        
        #----- Define N_des -----
        
        N_ref = list_of_N_des_coeff          
        N_des = np.array(len(N_ref)*[0.0]) 
            
        for i in range(len(N_ref)- 1):
            N_des[i] = np.floor(N_veh/np.sum(N_ref) * N_ref[i])
                      
        N_des[-1] = N_veh - np.sum(N_des) 
        
        gammas = np.load(folder_name + '/list_of_gammas.npy')
        
        if os.path.isfile(folder_name + '/fixed_prices.npy'):
            
            fixed_prices = np.load(folder_name + '/fixed_prices.npy')
            
        else:
            
            fixed_prices = None
        
        return Ag, bg, p_occ_belief, avg_earning_belief, Q, N_max, N_des, Map, RHVs, gammas, N_companies, fixed_prices
                