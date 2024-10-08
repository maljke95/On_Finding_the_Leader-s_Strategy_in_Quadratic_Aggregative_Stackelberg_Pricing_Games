# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 15:37:18 2022

@author: marko
"""

import numpy as np
from collections import deque

class Vehicle(object):
    
    def __init__(self, idx, start_node, allcoordinates, list_of_state_names, list_of_transitions, initial_state=None, distance_epsilon=0.000001):
        
        self.ID = idx
    
        self.previous_node = start_node
        self.previous_node_pos = np.array([allcoordinates[start_node, 0], allcoordinates[start_node, 1]])
        
        self.planned_route = deque([], maxlen=3000)
        
        self.next_node = None
        self.next_node_pos = None              
        self.remaining_distance_to_next_node = None
        
        self.x = self.previous_node_pos[0]
        self.y = self.previous_node_pos[1]
        self.traversed_distance = 0.0
                
        self.distance_epsilon = distance_epsilon
        
        #----- State Machine initial state -----
        
        self.handlers = {}
        self.current_state = initial_state
        
        self.setup(list_of_state_names, list_of_transitions)
        
        #----- Park the vehicle -----
        
        self.park = False
        
    #----- VEHICLE METHODS -----
    
    def plan_a_route_to_a_node(self, allpaths, destination_node, first_node=None):
        
        last_node = destination_node
    
        if first_node is None:
            
            first_node = self.previous_node            
            while not (last_node == first_node):
                
                next_node = last_node
                self.planned_route.appendleft(next_node)
                last_node = allpaths[int(first_node), int(last_node)]  
            
        else:
            
            route = deque([], maxlen=3000)
            while not (last_node == first_node):
                
                next_node = last_node
                route.appendleft(next_node)
                last_node = allpaths[first_node, last_node]  
                
            return route
        
    def get_coordinates(self, alldists):
        
        if not self.next_node is None:
            
            dl = alldists[self.previous_node, self.next_node] - self.remaining_distance_to_next_node
            pos = self.previous_node_pos + dl*(self.next_node_pos - self.previous_node_pos)/np.linalg.norm(self.next_node_pos - self.previous_node_pos)
            
        else:
            
            pos = self.previous_node_pos
            
        return np.array(pos)
    
    def update_coordinates(self, pos):
        
        self.x = pos[0]
        self.y = pos[1]        

    def move(self, v, dt, allcoordinates, alldists, vehicle_stopped=False):
        
        if not vehicle_stopped:
            
            ds = v*dt
            
            if self.remaining_distance_to_next_node - ds <= self.distance_epsilon:
                
                if self.remaining_distance_to_next_node >= ds:
                    
                    self.traversed_distance += v*dt
                    
                    self.previous_node = self.next_node
                    self.previous_node_pos = np.copy(self.next_node_pos)
                    
                    self.planned_route.popleft()
                    
                    if len(self.planned_route)>0:
                        
                        self.next_node = self.planned_route[0]
                        self.next_node_pos = np.array([allcoordinates[self.next_node, 0], allcoordinates[self.next_node, 1]])
                        self.remaining_distance_to_next_node = alldists[self.previous_node, self.next_node]
                        
                    else:
                        
                        self.next_node = None
                        self.next_node_pos = None
                        
                        self.remaining_distance_to_next_node = None 
                        
                else:
                    
                    start_node = self.planned_route[0]
                    previous_node = None
                    
                    while self.remaining_distance_to_next_node + alldists[start_node, self.planned_route[0]] < ds:
                        
                        previous_node = self.planned_route[0]
                        self.planned_route.popleft()
                        if len(self.planned_route) == 0:
                            break
                        
                    self.previous_node = previous_node
                    self.previous_node_pos = np.array([allcoordinates[previous_node, 0], allcoordinates[previous_node, 1]])
                    
                    if len(self.planned_route) == 0:
                        
                        self.traversed_distance += alldists[start_node, previous_node] + self.remaining_distance_to_next_node
                        
                        self.next_node = None
                        self.remaining_distance_to_next_node = None
                             
                    else:
                        
                        self.traversed_distance += ds
                        
                        self.next_node = self.planned_route[0]
                        self.next_node_pos = np.array([allcoordinates[self.next_node, 0], allcoordinates[self.next_node, 1]])
                        overshoot = ds - self.remaining_distance_to_next_node - alldists[start_node, previous_node]
                        self.remaining_distance_to_next_node = alldists[self.previous_node, self.planned_route[0]] - overshoot
                    
            else:
                
                self.traversed_distance += ds
                
                self.remaining_distance_to_next_node -= ds                               
                    
    def find_adjacent_intersection(self, curr_intersection, allpaths):
        
        list_of_adjacent_intersections = []
        N_intersect = allpaths.shape[1]
        
        for idx in range(N_intersect):
            if allpaths[curr_intersection, idx] == curr_intersection and not(curr_intersection == idx):
                list_of_adjacent_intersections.append(idx)
                
        return list_of_adjacent_intersections
    
    #----- STATE MACHINE METHODS -----

    def set_state(self, name):
        
        self.current_state = name
        
    def add_state(self, name, transition_function):
        
        self.handlers[name] = transition_function
            
    def setup(self, list_of_state_names, list_of_transitions):
        
        for i in range(len(list_of_state_names)):
            
            state_name = list_of_state_names[i]
            transition = list_of_transitions[i]
            self.add_state(state_name, transition)
                   
    def run_current_state(self):
        
        pass
        
    def transition_state(self, action):
        
        handler = self.handlers[self.current_state]
        self.current_state = handler(action)
                
    def update(self):
                
        action = self.run_current_state()
        self.transition_state(action)
        

                
            
        
        
        
        