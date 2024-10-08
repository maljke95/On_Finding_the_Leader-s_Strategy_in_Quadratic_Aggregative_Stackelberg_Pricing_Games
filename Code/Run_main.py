# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 20:13:26 2022

@author: marko
"""
import time
import os
from datetime import datetime 
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

from Simulator_electric import Simulator_electric
from Charging_balancing_modules.Game_s import Game

#----- Play an Inverse Stackelberg Game -----

def main_f_inverse_stackelberg():
    
    #----- Set the params of the simulator -----
    
    list_of_cID = [0, 1, 2]                                                     # List of company IDs
    fleet_sizes = [450, 400, 350]                                              # List of fleet sizes
    
    list_of_stations = [223, 7, 1102, 1723]                                     # Node IDs corresponding to charging stations
    list_of_capacities = [15, 60, 35, 50]                                       # Charging station Capacities
    
    t_end = 3.0
    
    #----- Run the simulator -----
    
    sim = Simulator_electric(t_start=0.0, batch_id=2, arrivals_id=0, list_of_stations = list_of_stations, list_of_capacities = list_of_capacities)
    sim.generate_simulator(list_of_cID, fleet_sizes)
    
    start_time = time.time()
    
    sim.run(t_end, additional_RHV_states_not_to_move=None)
    
    print("Execution time: ",time.time()-start_time)    
    
    #----- Run Inverse Stackelberg game -----
    
    current_folder = os.getcwd()+'/Results'
    if not os.path.isdir(current_folder):
        os.makedirs(current_folder)
        
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    name = current_folder + "/" + date_time

    if not os.path.isdir(name):
        os.makedirs(name)     
        
    print('')       
    print(10*'-'+' Game rebalancing '+10*'-')
        
    #----- Game -----
    
    fixed_params = {}
    fixed_params['fixed_exp_profit'] = None
    fixed_params['fixed_prices']  = None
    fixed_params['Q'] = 0.1*np.diag([4.0, 1.0, 3.0, 2.0])
    fixed_params['Ag'] = None
    fixed_params['N_des'] = None
    fixed_params['N_max'] = np.array(list_of_capacities)
    fixed_params['p_occ_belief'] = 3*[np.array([0.35, 0.1, 0.2, 0.15])]
    fixed_params['avg_earning_belief'] = None
    fixed_params['gammas'] = None
    fixed_params['bg'] = None
    
    start_time_of_games = time.time()
    
    g = Game() 
    
    Ag, bg, N_max, N_des, Q, p_occ_belief, avg_earning_belief, gammas = g.define_parameters(sim, fixed_params=fixed_params)
    government_performance_logger = g.Run_game_from_sim(Ag, bg, N_max, N_des, Q, p_occ_belief, avg_earning_belief, gammas, sim, fixed_params=fixed_params, plot_graphs=True, list_of_games=[0], list_of_games_to_play=[0], folder_name=name, robust_coeff=0.0, scale_param=1.0)
    
    execution_time = time.time() - start_time_of_games
    
    g.save_info_about_game(fixed_params, name)
    
    #government_performance_logger2, g2, execution_time2 = load_f_inverse_stackelberg(str(date_time))
    
    return government_performance_logger, sim, g, execution_time

#----- Load and play Inverse Stackelberg game -----

def load_f_inverse_stackelberg(name_string):
    
    #----- Run Inverse Stackelberg game -----
    
    current_folder = os.getcwd()+'/Results'
    name = current_folder + "/" + name_string  
        
    print('')       
    print(10*'-'+' Game rebalancing '+10*'-')
        
    #----- Game -----
    
    start_time_of_games = time.time()
    
    g = Game()
    
    Ag, bg, p_occ_belief, avg_earning_belief, Q, N_max, N_des, Map, RHVs, gammas, N_companies, fixed_price = g.load_info_about_game(name)
    government_performance_logger = g.Run_game_from_stored_data(Ag, bg, p_occ_belief, avg_earning_belief, Q, N_max, N_des, Map, RHVs, gammas, N_companies, fixed_price, plot_graphs=True, list_of_games=[0], list_of_games_to_play=[0], folder_name=name, robust_coeff=0.0, scale_param=1.0)
    
    execution_time = time.time() - start_time_of_games
    
    return government_performance_logger, g, execution_time

#----- Play a Stackelberg Game -----

def main_f_stackelberg():
    
    #----- Set the params of the simulator -----
    
    list_of_cID = [0, 1, 2]                                                     # List of company IDs
    fleet_sizes = [450, 400, 350]                                              # List of fleet sizes
    
    list_of_stations = [223, 7, 1102, 1723]                                     # Node IDs corresponding to charging stations
    list_of_capacities = [10, 60, 40, 45]                                       # Charging station Capacities
    
    t_end = 3.0
    
    #----- Run the simulator -----
    
    sim = Simulator_electric(t_start=0.0, batch_id=2, arrivals_id=0, list_of_stations = list_of_stations, list_of_capacities = list_of_capacities)
    sim.generate_simulator(list_of_cID, fleet_sizes)
    
    start_time = time.time()
    
    sim.run(t_end, additional_RHV_states_not_to_move=None)
    
    print("Execution time: ",time.time()-start_time)    
    
    #----- Run Stackelberg game -----
    
    current_folder = os.getcwd()+'/Results'
    if not os.path.isdir(current_folder):
        os.makedirs(current_folder)
        
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    name = current_folder + "/" + date_time

    if not os.path.isdir(name):
        os.makedirs(name)     
        
    print('')       
    print(10*'-'+' Game rebalancing '+10*'-')
    
    #----- Game -----
    
    fixed_params = {}
    fixed_params['fixed_exp_profit'] = None
    fixed_params['fixed_prices']  = np.array([5.0, 5.0, 5.0, 5.0])
    fixed_params['Q'] = 0.1*np.diag([5.0, 1.0, 3.0, 2.0])
    fixed_params['Ag'] = None
    fixed_params['N_des'] = None
    fixed_params['N_max'] = np.array(list_of_capacities)
    fixed_params['p_occ_belief'] = 3*[np.array([0.35, 0.1, 0.2, 0.15])]
    fixed_params['avg_earning_belief'] = None
    fixed_params['gammas'] = None
    fixed_params['bg'] = None
    
    start_time_of_games = time.time()
    
    g = Game()
    
    government_performance_logger = g.Run_game_from_sim(sim, fixed_params=fixed_params, plot_graphs=True, list_of_games=[1], list_of_games_to_play=[1], folder_name=name, robust_coeff=0.0, scale_param=1.0)
    
    g.save_info_about_game(fixed_params, name)
    
    execution_time = time.time() - start_time_of_games
    
    #government_performance_logger2, g2, execution_time2 = load_f_stackelberg(str(date_time))
    
    return government_performance_logger, sim, g, execution_time

def load_f_stackelberg(name_string):
    
    #----- Run Inverse Stackelberg game -----
    
    current_folder = os.getcwd()+'/Results'
    name = current_folder + "/" + name_string  
        
    print('')       
    print(10*'-'+' Game rebalancing '+10*'-')
        
    #----- Game -----
    
    start_time_of_games = time.time()
    
    g = Game()
    
    Ag, bg, p_occ_belief, avg_earning_belief, Q, N_max, N_des, Map, RHVs, gammas, N_companies, fixed_price = g.load_info_about_game(name)
    government_performance_logger = g.Run_game_from_stored_data(Ag, bg, p_occ_belief, avg_earning_belief, Q, N_max, N_des, Map, RHVs, gammas, N_companies, fixed_price, plot_graphs=True, list_of_games=[1], list_of_games_to_play=[1], folder_name=name, robust_coeff=0.0, scale_param=1.0)
    
    execution_time = time.time() - start_time_of_games
    
    return government_performance_logger, g, execution_time

#----- Robustness plots -----

def check_robustness_performance(name_string, fixed_price=None, save_figure=True):
    
    current_folder = os.getcwd()+'/Results'
    name = current_folder + "/" + name_string  
    
    #----- Game -----
    
    g1 = Game()
    g2 = Game()
    
    Ag, bg, p_occ_belief, avg_earning_belief, Q, N_max, N_des, Map, RHVs, gammas, N_companies, fixed_price_used = g1.load_info_about_game(name)
    
    scale_factors = [4.0]
    N_coeff = 6
    robust_coeffs = np.linspace(0.0, 1.0, N_coeff)
    N_iterations_per = 5

    results_pricing_policies = np.zeros((len(scale_factors), len(robust_coeffs), N_iterations_per))
    
    if not fixed_price is None:
        results_fixed_prices = np.zeros((len(scale_factors), len(robust_coeffs), N_iterations_per))
    
    iteration = 0
    
    for scale_id, scale_param in enumerate(scale_factors):
        for robust_id, robust_coeff in enumerate(robust_coeffs):
            for iter_id in range(N_iterations_per):
                
                print("Iteration: "+str(iteration))
                iteration += 1
    
                government_performance_logger1 = g1.Run_game_from_stored_data(Ag, bg, p_occ_belief, avg_earning_belief, Q, N_max, N_des, Map, RHVs, gammas, N_companies, fixed_price=None, plot_graphs=False, list_of_games=[0], list_of_games_to_play=[0], folder_name=name, robust_coeff=robust_coeff, scale_param=scale_param)
                results_pricing_policies[scale_id, robust_id, iter_id] = government_performance_logger1[0]
                g1.reset_game()
                
                if not fixed_price is None:
                    
                    government_performance_logger2 = g2.Run_game_from_stored_data(Ag, bg, p_occ_belief, avg_earning_belief, Q, N_max, N_des, Map, RHVs, gammas, N_companies, fixed_price, plot_graphs=False, list_of_games=[1], list_of_games_to_play=[1], folder_name=name, robust_coeff=robust_coeff, scale_param=scale_param)
                    results_fixed_prices[scale_id, robust_id, iter_id] = government_performance_logger2[0]  
                    g2.reset_game()
                    
    np.save(name + '/results_pricing_policies.npy', results_pricing_policies)
    if not fixed_price is None:
        np.save(name + '/results_fixed_prices.npy', results_fixed_prices)
        
    fig1, ax1 = plt.subplots(dpi=180)
    
    mean_policy = np.squeeze(np.mean(np.squeeze(results_pricing_policies[0, :, :]).T, axis=0))
    min_val_policy = np.squeeze(np.amin(np.squeeze(results_pricing_policies[0, :, :]).T, axis=0))
    max_val_policy = np.squeeze(np.amax(np.squeeze(results_pricing_policies[0, :, :]).T, axis=0))  
    
    if not fixed_price is None:
        
        mean_fixed = np.squeeze(np.mean(np.squeeze(results_fixed_prices[0, :, :]).T, axis=0))
        min_val_fixed = np.squeeze(np.amin(np.squeeze(results_fixed_prices[0, :, :]).T, axis=0))
        max_val_fixed = np.squeeze(np.amax(np.squeeze(results_fixed_prices[0, :, :]).T, axis=0))     
        
    ax1.plot(robust_coeffs, mean_policy,'-' ,label='Inverse Stackelberg') 
    
    if not fixed_price is None:

        ax1.plot(robust_coeffs, mean_fixed, '-', label='Stackelberg') 

    ax1.grid('on')
    ax1.legend()
    ax1.set_xlabel(r'$\alpha$')
    ax1.set_ylabel(r'$J_G(\sigma(x))$')

    tikzplotlib.clean_figure()
    tikzplotlib.save(name + "/Robust_plot.tex")
    
    if save_figure:
        fig1.savefig(name + "/Robust_plot.jpg", dpi=180)
        
    plt.show()
    plt.close()
    
def run_stackelberg(name_string, p):
    
    current_folder = os.getcwd()+'/Results'
    name = current_folder + "/" + name_string  
    
    g1 = Game()
    Ag, bg, p_occ_belief, avg_earning_belief, Q, N_max, N_des, Map, RHVs, gammas, N_companies, _ = g1.load_info_about_game(name)
    
    fixed_price = p
    government_performance_logger = g1.Run_game_from_stored_data(Ag, bg, p_occ_belief, avg_earning_belief, Q, N_max, N_des, Map, RHVs, gammas, N_companies, fixed_price, plot_graphs=False, list_of_games=[1], list_of_games_to_play=[1], folder_name=name, robust_coeff=0.0, scale_param=1.0)
    
    print(g1.list_of_GameCoord[0].Companies[0].xi)     
    print(g1.list_of_GameCoord[0].Companies[1].xi) 
    print(g1.list_of_GameCoord[0].Companies[2].xi)  
    print(government_performance_logger[0])
    print(g1.list_of_GameCoord[0].Companies[0].xi*194+g1.list_of_GameCoord[0].Companies[1].xi*181+g1.list_of_GameCoord[0].Companies[2].xi*157)
    
def find_optimal_fixed_prices(name_string, num_sample=10, n_repetitions = 0):

    current_folder = os.getcwd()+'/Results'
    name = current_folder + "/" + name_string  
    
    g1 = Game()
    Ag, bg, p_occ_belief, avg_earning_belief, Q, N_max, N_des, Map, RHVs, gammas, N_companies, _ = g1.load_info_about_game(name)
    
    p_min = 1.0
    p_max = 10.0
    
    p0_list = np.linspace(p_min, p_max, num=num_sample)
    p1_list = np.linspace(p_min, p_max, num=num_sample)
    p2_list = np.linspace(p_min, p_max, num=num_sample)
    p3_list = np.linspace(p_min, p_max, num=num_sample)
    
    iteration = 0
    
    p0_opt = 0.0
    p1_opt = 0.0
    p2_opt = 0.0
    p3_opt = 0.0
    
    optimal_loss = 10000000000000
    
    for p0_id, p0 in enumerate(p0_list):
        for p1_id, p1 in enumerate(p1_list):
            for p2_id, p2 in enumerate(p2_list):
                for p3_id, p3 in enumerate(p3_list):
                    
                    iteration += 1
                    
                    print('iteration number: '+str(iteration))
                    
                    fixed_price = np.array([p0, p1, p2, p3])

                    g1 = Game()
                    government_performance_logger = g1.Run_game_from_stored_data(Ag, bg, p_occ_belief, avg_earning_belief, Q, N_max, N_des, Map, RHVs, gammas, N_companies, fixed_price, plot_graphs=False, list_of_games=[1], list_of_games_to_play=[1], folder_name=name, robust_coeff=0.0, scale_param=1.0)
                    
                    print(government_performance_logger[0], optimal_loss, fixed_price)
                    if government_performance_logger[0] < optimal_loss:
                        
                        optimal_loss = government_performance_logger[0]
                        p0_opt = p0
                        p1_opt = p1
                        p2_opt = p2
                        p3_opt = p3
                        
    np.save(name + '/Optimal_fixed_prices.npy', np.array([p0_opt, p1_opt, p2_opt, p3_opt]))
    
    #----- Discretize better -----
    
    for rep in range(n_repetitions):
        
        opt_cost_init = optimal_loss
        
        p0_list = np.linspace(max(p_min, p0_opt - 1.0), min(p0_opt + 1.0, p_max), num_sample)
        p1_list = np.linspace(max(p_min, p1_opt - 1.0), min(p1_opt + 1.0, p_max), num_sample)
        p2_list = np.linspace(max(p_min, p2_opt - 1.0), min(p2_opt + 1.0, p_max), num_sample)
        p3_list = np.linspace(max(p_min, p3_opt - 1.0), min(p3_opt + 1.0, p_max), num_sample)
        
        for p0_id, p0 in enumerate(p0_list):
            for p1_id, p1 in enumerate(p1_list):
                for p2_id, p2 in enumerate(p2_list):
                    for p3_id, p3 in enumerate(p3_list):
                        
                        iteration += 1
                        
                        print('iteration number: '+str(iteration))
                        
                        fixed_price = np.array([p0, p1, p2, p3])
                               
                        g1 = Game()
                        government_performance_logger = g1.Run_game_from_stored_data(Ag, bg, p_occ_belief, avg_earning_belief, Q, N_max, N_des, Map, RHVs, gammas, N_companies, fixed_price, plot_graphs=False, list_of_games=[1], list_of_games_to_play=[1], folder_name=name, robust_coeff=0.0, scale_param=1.0)
                        
                        print(government_performance_logger[0], optimal_loss, fixed_price)
                        if government_performance_logger[0] < optimal_loss:
                            
                            optimal_loss = government_performance_logger[0]
                            p0_opt = p0
                            p1_opt = p1
                            p2_opt = p2
                            p3_opt = p3 
                            
        np.save(name + '/Optimal_fixed_prices.npy', np.array([p0_opt, p1_opt, p2_opt, p3_opt]))
        
        if opt_cost_init == optimal_loss:
            
            break
    
    print("The optimal fixed prices are: ", p0_opt, p1_opt, p2_opt, p3_opt)
    
    return p0_opt, p1_opt, p2_opt, p3_opt
        
if __name__ == '__main__': 
    
    # #----- Test both games -----
    
    # government_loss, sim, g, execution_time = main_f_inverse_stackelberg()
    # government_loss, sim, g, execution_time = main_f_stackelberg()
    
    # #----- Test assignment game -----
    
    # g.list_of_GameCoord[0].play_assignment_game(company_ID=0)
    
    # #----- Test robustness plot -----
    
    # name_string = '/03_17_2022_10_52_29'
    # fixed_price = np.array([5.0, 5.0, 5.0, 5.0])
    # check_robustness_performance(name_string, fixed_price, save_figure=True)
    
    #----- Test optimal fixed prices -----

    # name_string = '/03_23_2022_21_08_11'  
    # find_optimal_fixed_prices(name_string, num_sample = 2, n_repetitions = 3)
    
    #----- Find parameters for the ACC paper -----
    
    #name_string = '/03_23_2022_21_08_11'
    name_string = '/05_22_2022_00_31_41'
    
    government_performance_logger, g, execution_time = load_f_inverse_stackelberg(name_string)
    
    folder = os.getcwd() + '/Parameters_ACC'
    
    if not os.path.isdir(folder):
        os.makedirs(folder)
    
    N_des = g.list_of_GameCoord[0].N_des
    
    np.save(folder + '/N_des.npy', N_des)
    
    # for i in range(len(g.list_of_GameCoord[0].Companies)):
    
    #     Ni = g.list_of_GameCoord[0].Companies[i].cost_function.Ni
    
    #     Pi = g.list_of_GameCoord[0].Companies[i].cost_function.Ai/Ni**2
    #     Qi = g.list_of_GameCoord[0].Companies[i].cost_function.Bi/Ni
    #     ri = (g.list_of_GameCoord[0].Companies[i].cost_function.ci + g.list_of_GameCoord[0].Companies[i].cost_function.fi)/Ni
    #     Si = g.list_of_GameCoord[0].Companies[i].cost_function.Di/Ni
    
    #     K_l = g.list_of_GameCoord[0].Companies[i].K_l
    #     k_r = g.list_of_GameCoord[0].Companies[i].k_r*Ni
        
    #     Gi = np.concatenate((K_l[:4, :], K_l[6:, :]), axis=0)
    #     hi = np.concatenate((k_r[:4], k_r[6:]), axis=0)
        
    #     Ai = np.array([[1.0, 1.0, 1.0, 1.0]])
    #     bi = np.array([1.0])*Ni
        
        # np.save(folder + '/N'+str(i+1)+'.npy', Ni)
        # np.save(folder + '/P'+str(i+1)+'.npy', Pi)
        # np.save(folder + '/Q'+str(i+1)+'.npy', Qi)
        # np.save(folder + '/r'+str(i+1)+'.npy', ri)
        # np.save(folder + '/S'+str(i+1)+'.npy', Si)
        # np.save(folder + '/G'+str(i+1)+'.npy', Gi)
        # np.save(folder + '/h'+str(i+1)+'.npy', hi)
        # np.save(folder + '/A'+str(i+1)+'.npy', Ai)
        # np.save(folder + '/b'+str(i+1)+'.npy', bi)


    for i in range(len(g.list_of_GameCoord[0].Companies)):
    
        Ni = g.list_of_GameCoord[0].Companies[i].cost_function.Ni
    
        Pi = g.list_of_GameCoord[0].Companies[i].cost_function.Ai
        Qi = g.list_of_GameCoord[0].Companies[i].cost_function.Bi
        ri = g.list_of_GameCoord[0].Companies[i].cost_function.ci + g.list_of_GameCoord[0].Companies[i].cost_function.fi
        Si = g.list_of_GameCoord[0].Companies[i].cost_function.Di
    
        K_l = g.list_of_GameCoord[0].Companies[i].K_l
        k_r = g.list_of_GameCoord[0].Companies[i].k_r
        
        Gi = np.concatenate((K_l[:4, :], K_l[6:, :]), axis=0)
        hi = np.concatenate((k_r[:4], k_r[6:]), axis=0)
        
        Ai = np.array([[1.0, 1.0, 1.0, 1.0]])
        bi = np.array([1.0])
        
        np.save(folder + '/N'+str(i+1)+'.npy', Ni)
        np.save(folder + '/P'+str(i+1)+'.npy', Pi)
        np.save(folder + '/Q'+str(i+1)+'.npy', Qi)
        np.save(folder + '/r'+str(i+1)+'.npy', ri)
        np.save(folder + '/S'+str(i+1)+'.npy', Si)
        np.save(folder + '/G'+str(i+1)+'.npy', Gi)
        np.save(folder + '/h'+str(i+1)+'.npy', hi)
        np.save(folder + '/A'+str(i+1)+'.npy', Ai)
        np.save(folder + '/b'+str(i+1)+'.npy', bi)    
        
        np.save(folder + '/K_l'+str(i+1)+'.npy', K_l)  
        np.save(folder + '/k_r'+str(i+1)+'.npy', k_r) 
    
    p = np.array([3, 2, 2, 2])+np.random.rand(4)
    run_stackelberg(name_string, p)