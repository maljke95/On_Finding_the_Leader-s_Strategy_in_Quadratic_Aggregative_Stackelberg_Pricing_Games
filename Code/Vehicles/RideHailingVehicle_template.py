# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 17:51:08 2022

@author: marko
"""

from Vehicles.Vehicle_template import Vehicle

class RideHailingVehicle(Vehicle):
    
    def __init__(self, idx, start_node, allintersect, allcoordinates, cID, list_of_ride_hailing_vehicle_states, list_of_ride_hailing_vehicle_transitions, alldists, allpaths=None, capacity=1, initial_state=None, distance_epsilon=0.000001):
        
        super(RideHailingVehicle, self).__init__(idx, start_node, allcoordinates, list_of_ride_hailing_vehicle_states, list_of_ride_hailing_vehicle_transitions, initial_state, distance_epsilon)
        
        self.allpaths = allpaths
        self.allcoordinates = allcoordinates
        self.alldists = alldists
        
        self.list_of_adjacent_intersections = allintersect
        
        self.companyID = cID        
        self.capacity = capacity
        
        self.passengers_to_be_picked_up                = []
        self.origins_of_the_passengers_to_be_picked_up = []
        self.destinations_of_the_passengers_to_be_picked_up = []
                
        self.travelling_passengers                     = []
        self.destinations_of_the_travelling_passengers = []
        
    def update_allpaths(self, allpaths):
        
        self.allpaths = allpaths
        
    def assign_passenger(self, passengerID, origin, destination):
        
        self.passengers_to_be_picked_up.append(passengerID)
        self.origins_of_the_passengers_to_be_picked_up.append(origin)
        self.destinations_of_the_passengers_to_be_picked_up.append(destination)
        
    def passenger_picked_up(self, passengerID):
        
        self.travelling_passengers.append(passengerID)
        index = self.passengers_to_be_picked_up.index(passengerID)
        destination = self.destinations_of_the_passengers_to_be_picked_up[index]
        
        self.destinations_of_the_travelling_passengers.append(destination)
        
        self.passengers_to_be_picked_up.remove(passengerID)
        self.origins_of_the_passengers_to_be_picked_up.pop(index)
        self.destinations_of_the_passengers_to_be_picked_up.pop(index)
        
    def dispatch_passenger(self, passengerID):
        
        index = self.travelling_passengers.index(passengerID)
        self.travelling_passengers.remove(passengerID)
        self.destinations_of_the_travelling_passengers.pop(index) 
    
    #----- The states and the state handler -----
    
    def run_current_state(self):
        
        pass
    
    def run_idle_state(self):
        
        pass
    
    def run_pick_passenger_state(self):
        
        pass
    
    def run_transporting_passenger_state(self):
        
        pass
    
    def run_park_state(self):
        
        pass
    
    
               
       
            

                
                
                
                
                
                
                    

                
            
            
            
            
            
                
        
        
        