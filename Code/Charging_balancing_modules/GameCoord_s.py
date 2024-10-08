# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

class GameCoordinator():
    
    def __init__(self, idx=None, Map=None, Companies=None, Government=None, N_des=None, k_iter=3000):
        
        self.ID = idx
        
        self.Map = Map
        
        if Companies is None:
            self.Companies = []
        else:
            self.Companies = Companies
            
        self.Government = Government
        
        self.k_iter = k_iter
        
        self.N_des = N_des
        
        #----- Intermediate values -----
        
        self.sigma_x = None
        self.buffer_sigma_x = []
        self.list_of_welfare = []
        
    def set_map(self, Map):
        
        self.Map = Map
    
    def set_gov(self, Gov):
        
        self.Government = Gov
        
    def add_comp(self, c):
        
        self.Companies.append(c)
        
    def reset_buffers(self):
        
        self.list_of_welfare = []
                
    def collect_sigma_x(self):
        
        start_sum = np.squeeze(np.zeros((self.Map.M, 1)))
        
        for com in self.Companies:           
            start_sum += com.broadcast_decision()
            
        self.sigma_x = start_sum
        self.buffer_sigma_x.append(self.sigma_x) 
        
    def broadcast_sigma_x(self):
        
        for com in self.Companies:
            com.collect_sigma_x(self.sigma_x)
            
    def play_one(self):
        
        self.collect_sigma_x()
        
        prev_welfare = self.Government.calculate_welfare(self.sigma_x) + 0.5*self.N_des @ self.Government.Ag @ self.N_des
        self.list_of_welfare.append(prev_welfare)
        
        self.broadcast_sigma_x()
        
        for com in self.Companies:
            com.update_decision()
            
    def play(self):
        
        print("Playing game...")
        for i in range(self.k_iter):
            self.play_one()
        
    def plot_exp_accumulation_evolution(self, folder_name=None):
        
        fig1, ax1 = plt.subplots(dpi=180)
        
        k = np.arange(len(self.buffer_sigma_x))
        buffer_sigma_x = np.array(self.buffer_sigma_x)
        
        list_of_colors = ['tab:red', 'tab:orange', 'tab:gray', 'tab:cyan', 'tab:brown', 'tab:pink', 'tab:green', 'tab:olive', 'tab:purple']
        
        for i in range(self.Map.M):
            
            desired_value = self.N_des[i]
            color_ = list_of_colors[i]
            
            ax1.plot(k, np.squeeze(buffer_sigma_x[:, i]), color=color_, label='Station '+str(i+1))
            ax1.plot(k, len(k)*[desired_value],  '--', color=color_,)
        
        ax1.grid('on')
        ax1.legend()
        ax1.set_xlabel("Iteration [k]")
        ax1.set_ylabel("Expected vehicle accumulation")
        ax1.set_xlim((0, len(k)))
        
        plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=2)
        
        if not(folder_name is None):
            
            fig1.savefig(folder_name + "/Exp_acc_"+str(self.ID)+".jpg", dpi=180)
                
            #tikzplotlib.clean_figure()
            tikzplotlib.save(folder_name + "/Exp_acc_"+str(self.ID)+".tex")            
        
        #plt.show()   
        plt.close()
        
    def plot_NE_pricing(self, folder_name=None):
        
        prices = []
        list_of_markers = ["^", "s","p", "d",  "h", "X", "x", "*", "8", "1"]
        list_of_colors = ['tab:red', 'tab:orange', 'tab:gray', 'tab:cyan', 'tab:brown', 'tab:pink', 'tab:green', 'tab:olive', 'tab:purple']
        
        fig1, ax1 = plt.subplots(dpi=180)
        
        
        for com_id in range(len(self.Companies)):
            
            com = self.Companies[com_id]
            price_i = com.cost_function.calculate_prices(com.xi, com.sigma_x)
            price_i = np.squeeze(price_i)
            
            prices.append(price_i)
            
            for j in range(len(price_i)):
                
                if j == 0:
                    ax1.plot(j+1, price_i[j], color = list_of_colors[j], marker = list_of_markers[com_id], markersize = 10, label = 'Company '+str(com_id+1))
                    
                else:               
                    ax1.plot(j+1, price_i[j], color = list_of_colors[j], marker = list_of_markers[com_id], markersize = 10)
            
        prices = np.array(prices)
            
        ax1.grid(b='True', axis='x')
        
        ax1.set_xlabel('Station number')
        ax1.set_xticks([y + 1 for y in range(self.Map.M)])
        fig1.suptitle('Charging price distribution at each station')
        
        frame1 = plt.gca()
        frame1.axes.xaxis.set_ticklabels([])
        
        plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
        
        if not(folder_name is None):
            
            fig1.savefig(folder_name + "/NE_prices_"+str(self.ID)+".jpg", dpi=180)
                
            #tikzplotlib.clean_figure()
            tikzplotlib.save(folder_name + "/NE_prices_"+str(self.ID)+".tex")  
            np.save(folder_name + "/prices_"+str(self.ID)+".npy", prices)
            
        #plt.show()  
        plt.close()
        
    def plot_losses(self, folder_name=None):
        
        fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=180)
        
        k = np.arange(self.k_iter)
        ax1.plot(k, self.list_of_welfare)
        ax1.grid('on')
        fig1.suptitle('Global cost init period')
        ax1.set_xlabel('Iteration [k]')
        ax1.set_ylabel(r'$J_{G}(x)$')
        ax1.set_xlim(1, 500)

        if not(folder_name is None):
            
            fig1.savefig(folder_name + "/Global_cost_init_period_"+str(self.ID)+".jpg", dpi=180)
                
            #tikzplotlib.clean_figure()
            tikzplotlib.save(folder_name + "/Global_cost_init_period_"+str(self.ID)+".tex") 

        fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=180)
        
        counter = 0
        for com in self.Companies:
            k = np.arange(len(com.loss_buffer))
            ax2.plot(k, np.array(com.loss_buffer)/com.Ni**2, label = 'Company '+str(counter+1))
            counter += 1
        
        ax2.grid('on')
        ax2.set_xlabel('Iteration [k]')
        ax2.set_ylabel(r'$\frac{1}{N_{i}^{2}}\: J^{i}(x_{i},x_{-i})$')
        ax2.set_xlim(1, 1000)
        ax2.legend(loc='lower right')
        fig2.suptitle('Company loss functions')

        if not(folder_name is None):
            
            fig2.savefig(folder_name + "/Company_loss_functions_"+str(self.ID)+".jpg", dpi=180)
                
            #tikzplotlib.clean_figure()
            tikzplotlib.save(folder_name + "/Company_loss_functions_"+str(self.ID)+".tex") 

        fig3, ax3 = plt.subplots(figsize=(6, 4), dpi=180)
        
        k = np.arange(self.k_iter)
        ax3.plot(k, self.list_of_welfare, 'k')
        ax3.grid('on')
        fig3.suptitle('Global cost whole length')
        ax3.set_xlabel('Iteration [k]')
        ax3.set_xlim(1, self.k_iter)
        
        if not(folder_name is None):
            
            fig3.savefig(folder_name + "/Global_cost_whole_length_"+str(self.ID)+".jpg", dpi=180)
                
            #tikzplotlib.clean_figure()
            tikzplotlib.save(folder_name + "/Global_cost_whole_length_"+str(self.ID)+".tex")  
        
        #plt.show()  
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        
    # def play_assignment_game(self, company_ID):
        
    #     v_avr = 20.0
    #     Tv = 2
        
    #     x_final1, e_final1, suma1, N_des_com1 = self.Companies[company_ID].assign_vehicles_to_charging_stations_single_surge_pricing(v_avr, Tv)
    #     x_final2, e_final2, suma2, N_des_com2, list_of_surges2 = self.Companies[company_ID].assign_vehicles_to_charging_stations_individual_surge_pricing(v_avr, Tv)
        
    #     if np.linalg.norm(suma1 - N_des_com1) > 0.0001:  
            
    #         single_surge = False  
            
    #     else:
            
    #         if np.max(np.max(list_of_surges2, axis=1)) < np.max(np.max(e_final1)):         
    #             single_surge = False
            
    #         else:           
    #             single_surge = True
            
    #     if single_surge:           
    #         print('Surge used: ', e_final1)
            
    #     else:            
    #         print('Max Surges used: ', np.max(np.max(list_of_surges2, axis=1)))
            
        
        
            
            