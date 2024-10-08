# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 20:08:39 2022

@author: marko
"""

import numpy as np

class MatchingAlgorithm(object):
    
    def __init__(self, max_cost=1000000):
        
        self.cost_matrix = None
        
        self.max_cost = max_cost
        
        #----- Result -----
        
        self.assignment_matrix = None

    def setup(self, t_curr, vehicles, passenger_ids, passenger_nodes, passenger_dest, request_times, max_waiting_times, avg_vels, alldists):
        
        N_veh = len(vehicles)
        N_pass = len(passenger_nodes)
        
        self.cost_matrix = self.max_cost*np.ones((N_pass, N_veh))
        
        for veh_id in range(len(vehicles)):
            
            veh = vehicles[veh_id]
            avg_vel = avg_vels[veh_id]
            
            next_node = veh.next_node
            remaining_distance = veh.remaining_distance_to_next_node
            
            if next_node is None:
                
                remaining_distance = 0.0
                next_node = veh.previous_node
                
            for passenger_id in range(len(passenger_ids)):
                
                passenger_node = passenger_nodes[passenger_id]
                request_time   = request_times[passenger_id]
                max_waiting    = max_waiting_times[passenger_id]
                dist = remaining_distance + alldists[next_node, passenger_node]

                Dt = dist/avg_vel
                
                waiting = t_curr - request_time + Dt
                if waiting <= max_waiting:
                    
                    self.cost_matrix[passenger_id, veh_id] = waiting 
                    
                    #----- sanity check -----
                    #assert(waiting<self.max_cost) 
                    
    def match(self, N_veh, N_pass):

        pass
        
    def assign_passengers(self, t_curr, vehicles, vehicle_capacities, passenger_ids, passenger_nodes, passenger_dest, request_times, max_waiting_times, avg_vels, alldists):

        N_pass = len(passenger_nodes) 
        N_veh = len(vehicles)
        
        self.vehicle_capacities = vehicle_capacities
        
        self.setup(t_curr, vehicles, passenger_ids, passenger_nodes, passenger_dest, request_times, max_waiting_times, avg_vels, alldists)

        self.match(N_veh, N_pass)

        list_of_not_assigned = []
        
        for idx in range(N_pass):
            
            passengerID = passenger_ids[idx]
            origin      = passenger_nodes[idx]
            destination = passenger_dest[idx]  
            
            if np.sum(self.assignment_matrix[idx, :] == 1):
                
                assigned_veh = np.where(self.assignment_matrix[idx, :] == 1)
                assigned_veh = assigned_veh[0][0]

                if self.cost_matrix[idx, assigned_veh] < self.max_cost:
                
                    vehicles[assigned_veh].assign_passenger(passengerID, origin, destination)
                    vehicles[assigned_veh].update()
                
                else:
                
                    element = {'passengerID':passengerID, 'origin':origin, 'destination':destination}
                    list_of_not_assigned.append(element)
                    
            else:
                
                element = {'passengerID':passengerID, 'origin':origin, 'destination':destination}
                list_of_not_assigned.append(element)                
                
        print("N_pass, N_veh, N_not: ", N_pass, len(vehicles), len(list_of_not_assigned))
                
        return list_of_not_assigned               
                