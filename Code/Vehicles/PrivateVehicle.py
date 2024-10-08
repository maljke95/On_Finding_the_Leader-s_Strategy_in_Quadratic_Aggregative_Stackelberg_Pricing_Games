# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 23:34:42 2022

@author: marko
"""
import numpy as np

from Vehicles.Vehicle_template import Vehicle

class PrivateVehicle(Vehicle):
    
    def __init__(self, idx, start_node, destination_node, allcoordinates, alldists, allpaths, list_of_private_vehicle_states, list_of_private_vehicle_transitions, initial_state="transporting_passenger", distance_epsilon=0.000001):
        
        super(PrivateVehicle, self).__init__(idx, start_node, allcoordinates, list_of_private_vehicle_states, list_of_private_vehicle_transitions, initial_state, distance_epsilon)
       
        self.plan_a_route_to_a_node(allpaths, destination_node)
        
        self.next_node = self.planned_route[0]
        self.next_node_pos = np.array([allcoordinates[self.next_node, 0], allcoordinates[self.next_node, 1]])
        self.remaining_distance_to_next_node = alldists[self.previous_node, self.next_node]
        
    def run_current_state(self):
                   
        if self.current_state == "transporting_passenger":            
            return self.run_transporting_passenger_state()         
        if self.current_state == "park":
            return self.run_park_state()
        
    def run_transporting_passenger_state(self):
        
        if self.next_node is None:
            
            action = 1
            
        else:
            
            action = 0
            
        return action
    
    def run_park_state(self):
        
        action = 0
        
        return action   