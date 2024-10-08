# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 10:55:28 2022

@author: marko
"""
import numpy as np

from Vehicles.PrivateVehicle import PrivateVehicle
from Vehicles.State_transitions.States_private_veh import f_transporting_passenger, f_park

from Matching_modules.MatchingModule_1 import MatchingAlgorithm_1 as MatchingAlgorithm

class Simulator_template():
    
    def __init__(self, t_start=0.0):
        
        self.t = t_start
        
        #----- MAP -----
        self.Map = None
        
        #----- Simulation sampling period -----
        
        self.dt = None
        
        #----- VEHICLES -----
        
        self.N_companies = 0
        self.RHVs = []
        self.n_idle_rhvs = 0
        
        self.PVs  = []
        
        #----- MATCHING -----
        
        self.matching_module = None
        
        #----- Matching sampling period -----
        
        self.k_match = None
        
        #----- Info log -----
        
        self.n_requests = 0
        self.n_served = 0
        self.n_abandoned = 0
        
        self.nV = 0.0
                       
    def setup_map(self):
        
        pass
    
    #----- Standard RHV -----
    
    def generate_RHV(self):
        
        pass
    
    def add_RHVs(self, lro, lri, list_of_cID):
        
        pass

    def prepare_RHV(self, list_of_cID, fleet_sizes):
        
        pass
    
    #----- PV -----
    def generate_PV(self, idx, start_node, destination_node, allcoordinates, alldists, allpaths, distance_epsilon=0.000001):
        
        list_of_private_vehicle_states = ["transporting_passenger", "park"]
        list_of_private_vehicle_transitions = [f_transporting_passenger, f_park]
        initial_state = "transporting_passenger"
        
        veh = PrivateVehicle(idx, start_node, destination_node, allcoordinates, alldists, allpaths, list_of_private_vehicle_states, list_of_private_vehicle_transitions, initial_state, distance_epsilon)
        
        return veh
        
    def add_PVs(self, lpo, lpd, lpi):
        
        allcoordinates = self.Map.coordinates
        allpaths = self.Map.allpaths  
        
        alldists = self.Map.alldists
        distance_epsilon = 0.000001
        
        for idx in range(len(lpi)):
            
            veh_id = lpi[idx]
            veh_origin = lpo[idx]
            veh_destination = lpd[idx]
            
            veh = self.generate_PV(veh_id, veh_origin, veh_destination, allcoordinates, alldists, allpaths, distance_epsilon)
            self.PVs.append(veh)
            
        self.nV += len(lpi)
    
    #----- All vehicles -----
    
    def update_all_vehicles(self):
        
        for rhv in self.RHVs:
            rhv.update()
            
        for pv in self.PVs:
            pv.update()
            
    def move_vehicles(self, v, dt, list_of_PV_states_not_to_move, list_of_RHV_states_not_to_move):
        
        new_list = []

        for pv in self.PVs:
            
            if not (pv.current_state in list_of_PV_states_not_to_move) and not (pv.remaining_distance_to_next_node is None):
                pv.move(v, dt, self.Map.coordinates, self.Map.alldists)

            pv.update()
            if pv.current_state == "park":
                self.nV -= 1
            else:
                new_list.append(pv)
        
        self.PVs = new_list
        
        
        n_idle_rhvs = 0
        
        for rhv in self.RHVs:
            
            if not (rhv.current_state in list_of_RHV_states_not_to_move) and not (rhv.remaining_distance_to_next_node is None):
                rhv.move(v, dt, self.Map.coordinates, self.Map.alldists)
                            
            rhv.update()
            
            if rhv.current_state == "idle":
                n_idle_rhvs += 1
                
        self.n_idle_rhvs = n_idle_rhvs

    def setup_matching(self):
        
        self.matching_module = MatchingAlgorithm(max_cost=1000000)
        self.k_match = 3

    def perform_matching(self, t_previous_assignment):
        
        list_of_arrtime, list_of_origins, list_of_dest, list_of_trip, list_of_ids = self.Map.fetch_requests(t_previous_assignment, self.t)
        lpa, lpo, lpd, lpt, lpi, lra, lro, lrd, lrt, lri = self.Map.sample_private_and_ride_hailing_requests(list_of_arrtime, list_of_origins, \
                                                            list_of_dest, list_of_trip, list_of_ids,  prob=0.2)
        
        self.n_requests += len(lra)  
        
        #----- Add immediate private requests -----
        
        self.add_PVs(lpo, lpd, lpi)

        #----- Matching parameters -----
        
        t_curr = self.t
        
        vehicles = []
        
        for veh in self.RHVs:
            if veh.current_state == 'idle':
                vehicles.append(veh)
        
        vehicle_capacities = np.array(len(vehicles)*[1.0])
        passenger_ids = lri
        passenger_nodes = lro
        passenger_dest  = lrd
        request_times   = lra
        max_waiting_times = len(passenger_ids)*[0.2]
        
        
        avg_vels = len(vehicles)*[self.Map.vel(self.nV)]
        
        list_of_not_assigned = self.matching_module.assign_passengers(t_curr, vehicles, vehicle_capacities, passenger_ids, passenger_nodes, passenger_dest, request_times, max_waiting_times, avg_vels, self.Map.alldists)
        
        self.n_served += len(lra) - len(list_of_not_assigned)
        
        #----- Add failed ridehailing requests to private vehicles -----
        
        if len(list_of_not_assigned):
            
            lpo2 = []
            lpd2 = []
            lpi2 = []
            
            for elem in list_of_not_assigned:
                
                lpo2.append(elem['origin'])
                lpd2.append(elem['destination'])
                lpi2.append(elem['passengerID'])
            
            self.add_PVs(lpo2, lpd2, lpi2)
            self.n_abandoned += len(list_of_not_assigned)
            
    def generate_simulator(self, list_of_cID, fleet_sizes):
        
        #----- Generate RHVs in the sim -----
        
        assert(len(list_of_cID) == len(fleet_sizes))
        
        self.prepare_RHV(list_of_cID, fleet_sizes)
        
        #----- Add matching -----
        
        self.setup_matching()
        
    def run(self, t_end, additional_RHV_states_not_to_move=None):
        
        t_previous_assignment = self.t
        
        #----- Set states not to be moved for PVs and RHVs ----
        
        list_of_PV_states_not_to_move = ["park"]
        list_of_RHV_states_not_to_move = ["park"]
        
        if additional_RHV_states_not_to_move is None:
            additional_RHV_states_not_to_move = []
            
        list_of_RHV_states_not_to_move += additional_RHV_states_not_to_move
        
        self.t += self.dt
        
        #----- Simulation -----
        
        self.update_all_vehicles()
        
        while self.t <= t_end:
            
            print("current time: "+"{0: .3f}".format(self.t))
            
            if self.t >= t_previous_assignment + self.k_match*self.dt and self.t<=t_end:
                
                #----- Perform vehicle matching and update their state -----

                self.perform_matching(t_previous_assignment)
                t_previous_assignment = self.t               
          
            #----- Move vehicles and updatet their state -----
            
            v = self.Map.vel(self.nV)
            
            print("Vehicles in the network: ",self.nV)
            print("N Idle RHVs: ",self.n_idle_rhvs)
            print("Velocity: ", v)
            
            self.move_vehicles(v, self.dt, list_of_PV_states_not_to_move, list_of_RHV_states_not_to_move)
                    
            self.t += self.dt
            print(30*'-')