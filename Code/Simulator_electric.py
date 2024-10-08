# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 14:29:25 2022

@author: marko
"""

import os
import numpy as np
import time

from Simulator_template import Simulator_template

from Maps.MapShenzhen_electric import MapShenzhen_electric

from Vehicles.RideHailingVehicle_electric import RideHailingVehicle_electric
from Vehicles.State_transitions.States_ride_hailing_veh import f_idle_electric, f_pick_passenger, f_transporting_passenger, f_park, f_charge

class Simulator_electric(Simulator_template):
    
    def __init__(self, t_start=0.0, batch_id=2, arrivals_id=0, list_of_stations = None, list_of_capacities = None):
        
        super(Simulator_electric, self).__init__(t_start)
        
        self.setup_map(batch_id, arrivals_id, list_of_stations, list_of_capacities)
        
    def setup_map(self, batch_id=2, arrivals_id=0, list_of_stations=None, list_of_capacities=None):
        
        path = os.getcwd()
        
        path_data           = path + '/Maps/Data/input_data.mat'
        path_arrivals       = path + '/Maps/Data/arrival batchs/arrivals_40k80k.mat'
        path_coordinates    = path + '/Maps/Data/Grid/coordinates.mat'
        
        self.Map = MapShenzhen_electric(batch_id, arrivals_id, list_of_stations, list_of_capacities, path_data, path_arrivals, path_coordinates)
        self.Map.setup_demand_based_values()
        
        self.dt = self.Map.tstep
        
    def generate_RHV(self, idx, start_node, allintersect, allcoordinates, cID, alldists, allpaths=None, capacity=1, distance_epsilon=0.000001):
        
        list_of_ride_hailing_vehicle_states = ["idle", "pick_passenger", "transporting_passenger", "park", "charge"]
        list_of_ride_hailing_vehicle_transitions = [f_idle_electric, f_pick_passenger, f_transporting_passenger, f_park, f_charge]
        initial_state = "idle"
        
        current_battery_level = np.random.uniform(90.0, 95.0, 1)
        max_range = np.random.uniform(150.0, 200.0, 1)
        charge_thr = np.random.uniform(55.0, 60.0, 1)
        desired_battery_level = np.random.uniform(96.0, 100.0, 1)
        
        veh = RideHailingVehicle_electric(idx, start_node, allintersect, allcoordinates, cID, list_of_ride_hailing_vehicle_states, list_of_ride_hailing_vehicle_transitions, alldists, charge_thr, max_range, current_battery_level, desired_battery_level, allpaths, capacity, initial_state, distance_epsilon)
        
        return veh
    
    def add_RHVs(self, lro, lri, list_of_cID):
        
        allcoordinates = self.Map.coordinates
        alldists = self.Map.alldists
        allpaths = self.Map.allpaths
        allintersect = self.Map.allintersect
        capacity = 1
        distance_epsilon=0.000001
             
        for idx in range(len(lri)):
            
            veh_id = lri[idx]
            veh_origin = lro[idx]
            cID = list_of_cID[idx]
            
            veh = self.generate_RHV(veh_id, veh_origin, allintersect, allcoordinates, cID, alldists, allpaths, capacity, distance_epsilon)
            self.RHVs.append(veh)
        
        self.n_idle_rhvs += len(lri) 
        self.nV += len(lri)
        
    def prepare_RHV(self, list_of_cID, fleet_sizes):
        
        #----- Generate RHVs in the sim -----
        
        self.N_companies = len(list_of_cID)
        
        first_id = 0
        for fleet_size_id in range(len(fleet_sizes)):
            
            cID = list_of_cID[fleet_size_id]
            
            fleet_size = fleet_sizes[fleet_size_id]
            list_of_cID_fleet = fleet_size * [cID]
            
            lro = np.random.randint(0, len(self.Map.alldists)-1, fleet_size)
            lri = list(np.arange(first_id, first_id + fleet_size))
            first_id += fleet_size
            
            self.add_RHVs(lro, lri, list_of_cID_fleet)
            
if __name__ == '__main__':        
            
    list_of_cID = [0, 1, 2]                                                     # List of company IDs
    fleet_sizes = [800, 250, 200]                                              # List of fleet sizes
    
    list_of_stations = [223, 7, 1102, 1723]                                     # Node IDs corresponding to charging stations
    list_of_capacities = [10, 60, 40, 45]                                       # Charging station Capacities
    
    t_end = 3.0
      
    sim = Simulator_electric(t_start=0.0, batch_id=2, arrivals_id=0, list_of_stations = list_of_stations, list_of_capacities = list_of_capacities)
    sim.generate_simulator(list_of_cID, fleet_sizes)
    
    start_time = time.time()
    
    sim.run(t_end, additional_RHV_states_not_to_move=None)
    
    print("Execution time: ",time.time()-start_time)