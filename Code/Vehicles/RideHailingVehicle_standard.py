# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 10:25:51 2022

@author: marko
"""

import numpy as np
import random
from collections import deque

from Vehicles.RideHailingVehicle_template import RideHailingVehicle

class RideHailingVehicle_standard(RideHailingVehicle):
    
    def __init__(self, idx, start_node, allintersect, allcoordinates, cID, list_of_ride_hailing_vehicle_states, list_of_ride_hailing_vehicle_transitions, alldists, allpaths=None, capacity=1, initial_state=None, distance_epsilon=0.000001):
        
        super(RideHailingVehicle_standard, self).__init__(idx, start_node, allintersect, allcoordinates, cID, list_of_ride_hailing_vehicle_states, list_of_ride_hailing_vehicle_transitions, alldists, allpaths, capacity, initial_state, distance_epsilon)
        
    def run_current_state(self):
        
        if self.current_state == "idle":            
            return self.run_idle_state()
        if self.current_state == "pick_passenger":            
            return self.run_pick_passenger_state()            
        if self.current_state == "transporting_passenger":            
            return self.run_transporting_passenger_state()         
        if self.current_state == "park":
            return self.run_park_state()
            
    def run_idle_state(self):
        
        if len(self.passengers_to_be_picked_up)>0:
            
            action = 1
            
            if self.next_node is None:
                
                if self.previous_node == self.origins_of_the_passengers_to_be_picked_up[0]:
                    
                    self.next_node = None
                    self.next_node_pos = None
                    self.remaining_distance_to_next_node = None
                                      
                else:
                    
                    self.plan_a_route_to_a_node(self.allpaths, destination_node=self.origins_of_the_passengers_to_be_picked_up[0])
                    self.next_node = self.planned_route[0]
                    self.next_node_pos = np.array([self.allcoordinates[self.next_node, 0], self.allcoordinates[self.next_node, 1]])
                    self.remaining_distance_to_next_node = self.alldists[self.previous_node, self.next_node]
                
            else:
                
                route = self.plan_a_route_to_a_node(self.allpaths, destination_node=self.origins_of_the_passengers_to_be_picked_up[0], first_node=self.next_node)
                route.appendleft(self.next_node)
                
                self.planned_route = route
                
        elif self.park:
            
            action = 100
            
        else:
            
            if self.next_node is None:
                
                self.next_node = random.choice(self.list_of_adjacent_intersections[self.previous_node])
                self.planned_route.append(self.next_node)
                self.next_node_pos = np.array([self.allcoordinates[self.next_node, 0], self.allcoordinates[self.next_node, 1]])
                self.remaining_distance_to_next_node = self.alldists[self.previous_node, self.next_node]
                
            action = 0
                                
        return action
               
    def run_pick_passenger_state(self):
        
        if self.next_node is None:                                              # This means the vehicle has arrived at the origin of the passenger
            
            action = 1
            
            passengerID = self.passengers_to_be_picked_up[0]
            
            self.passenger_picked_up(passengerID)
            index = self.travelling_passengers.index(passengerID)
            
            destination = self.destinations_of_the_travelling_passengers[index]
            self.plan_a_route_to_a_node(self.allpaths, destination_node=destination)
            self.next_node = self.planned_route[0]
            self.next_node_pos = np.array([self.allcoordinates[self.next_node, 0], self.allcoordinates[self.next_node, 1]])
            self.remaining_distance_to_next_node = self.alldists[self.previous_node, self.next_node]
            
        else:
            
            action = 0
            
        return action
    
    def run_transporting_passenger_state(self):
        
        if self.next_node is None:
            
            action = 1
            
            passengerID = self.travelling_passengers[0]
            
            self.dispatch_passenger(passengerID)
            self.next_node = None
            self.remaining_distance_to_next_node = None
            self.planned_route = deque([], maxlen=3000)
            
        else:
            
            action = 0
            
        return action
    
    def run_park_state(self):
        
        action = 0
        
        return action       

