# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 21:11:00 2022

@author: marko
"""
import numpy as np
import random
from collections import deque

from Vehicles.RideHailingVehicle_template import RideHailingVehicle

class RideHailingVehicle_electric(RideHailingVehicle):
    
    def __init__(self, idx, start_node, allintersect, allcoordinates, cID, list_of_ride_hailing_vehicle_states, list_of_ride_hailing_vehicle_transitions, alldists, charge_thr=None, max_range=None, current_battery_level=None, desired_battery_level=None, allpaths=None, capacity=1, initial_state=None, distance_epsilon=0.000001):
        
        super(RideHailingVehicle_electric, self).__init__(idx, start_node, allintersect, allcoordinates, cID, list_of_ride_hailing_vehicle_states, list_of_ride_hailing_vehicle_transitions, alldists, allpaths, capacity, initial_state, distance_epsilon)
        
        if current_battery_level is None:
            self.current_battery_level = np.random.uniform(80.0, 95.0)
        else:
            self.current_battery_level = current_battery_level
            
        if max_range is None:
            self.max_range = np.random.uniform(150.0, 200.0)
        else:
            self.max_range = max_range
            
        if charge_thr is None:
            self.charge_thr = np.random.uniform(40.0, 45.0)
        else:
            self.charge_thr = charge_thr
        
        if desired_battery_level is None:
            self.desired_battery_level = np.random.uniform(95.0, 100.0)
        else:
            self.desired_battery_level = desired_battery_level
        
        #----- Parameters for choosing where to go and charge -----
        
        self.list_of_station_ID = None
        self.dist = None        
        self.mi = None
        self.feasible_stations = []
        
    def run_current_state(self):
        
        if self.current_state == "idle":            
            return self.run_idle_state()
        if self.current_state == "pick_passenger":            
            return self.run_pick_passenger_state()            
        if self.current_state == "transporting_passenger":            
            return self.run_transporting_passenger_state()        
        if self.current_state == "charge":            
            return self.run_charge_state()  
        if self.current_state == "park":
            return self.run_park_state()
            
    def run_idle_state(self):

        if self.current_battery_level[0] <= self.charge_thr[0]:
            
            action = 2
            
        elif len(self.passengers_to_be_picked_up)>0:
            
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
    
    def run_charge_state(self):
        
        action = 0
        
        if self.park:
            
            action = 1
        
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
    
    #----- Redefined move method from class Vehicle -----
    
    def move(self, v, dt, allcoordinates, alldists, vehicle_stopped=False):
        
        previous_traversed_distance = self.traversed_distance
        
        super(RideHailingVehicle_electric, self).move(v, dt, allcoordinates, alldists, vehicle_stopped)
        
        ds = self.traversed_distance - previous_traversed_distance
        self.current_battery_level -= 100.0/self.max_range*ds
        
    #----- Calculate the params that influence charging station decision -----
    
    def update_vehicle_for_charging(self, stations):
        
        self.list_of_station_ID = stations
        
        dist = []
        
        if self.next_node is None:
            next_intersection = self.previous_node
            ds = 0.0
        else:
            next_intersection = self.next_node
            ds = self.remaining_distance_to_next_node
        
        for i in range(len(stations)):
            
            station_id = stations[i]
            dist.append(self.alldists[next_intersection, station_id] + ds)
            
            if self.current_battery_level - 100.0/self.max_range*dist[i] >= 0.0:
                
                    self.feasible_stations.append(i)
                    
        self.mi = len(self.feasible_stations)
        self.dist = np.array(dist)
        
    #----- Prepare for the surge pricing assignement procedure -----
    
    def prepare_assignment_procedure_for_vehicle(self, Tv, p_occ, avg_earning, exp_profit):
        
        feasible_set = []
        infeasible_set = []
        
        M = len(self.list_of_station_ID)
        
        ch_feas = np.zeros(M)
        
        for station_id, station in enumerate(self.feasible_stations):
            
            x_f = np.zeros(M)
            x_f[station] = 1
            
            if self.current_battery_level - 100.0/self.max_range*self.dist[station_id] >= 0.0 : 
                
                feasible_set.append(x_f)
                ch_feas[station] = self.desired_battery_level - (self.current_battery_level - 100.0/self.max_range*self.dist[station_id])
            
            else:
                
                infeasible_set.append(x_f)
                
        Dv = np.diag(ch_feas)
        
        e_arr = avg_earning * np.diag(p_occ) @ np.array(self.dist) 
        e_pro = np.array(exp_profit)*Tv/8
        
        neg_exp_revenue = e_arr - e_pro
        
        if len(infeasible_set)>0:
            
            Kl_a = np.array(infeasible_set)
            kr_a = np.zeros(len(infeasible_set))
            
        else:
            
            Kl_a = None
            kr_a = None
             
        return Dv, neg_exp_revenue, Kl_a, kr_a, feasible_set
        
        
        
