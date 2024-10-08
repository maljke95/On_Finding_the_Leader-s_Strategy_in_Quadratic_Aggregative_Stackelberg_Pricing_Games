# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 15:37:47 2022

@author: marko
"""
import numpy as np

from Maps.MapShenzhen import MapShenzhen

class MapShenzhen_electric(MapShenzhen):
    
    def __init__(self, batch_id=2, arrivals_id=0, list_of_stations=None, list_of_capacities=None, path_data=None, path_arrivals=None, path_coordinates=None):
        
        super(MapShenzhen_electric, self).__init__(batch_id, arrivals_id, path_data, path_arrivals, path_coordinates)
        
        if list_of_stations is None:
            
            self.stations = []
            self.N_max = []
            
        else:
            
            self.stations = list_of_stations
            self.N_max = list_of_capacities

        self.M = len(self.stations)
        
        self.ExpProfit = None
        self.N_des_coeff = None
        
    def setup_demand_based_values(self, ExpProfit = None, N_des_coeff = None, scale_factor = 800.0):
        
        if not(ExpProfit is None) and not(N_des_coeff is None):
            
            self.ExpProfit = ExpProfit
            self.N_des_coeff = N_des_coeff
            
        else:
            
            number_of_nodes = len(self.number_of_occ_origin)
            number_of_occ = self.number_of_occ_origin
                
            N_centr = len(self.stations)
            
            request_matrix = np.zeros(N_centr)
            node_centroid = []
            
            for node_id in range(number_of_nodes):
                
                node_x = self.coordinates[node_id, 0]
                node_y = self.coordinates[node_id, 1]
                
                closest_centr = None
                dist = None
                
                for centroid_id in range(N_centr):
                    
                    centroid = self.stations[centroid_id]
                    
                    centroid_x = self.coordinates[centroid, 0]
                    centroid_y = self.coordinates[centroid, 1]
                    
                    if closest_centr is None:
                        
                        dist = ((node_x - centroid_x)**2 + (node_y - centroid_y)**2)**0.5
                        closest_centr = centroid_id
                        
                    elif ((node_x - centroid_x)**2 + (node_y - centroid_y)**2)**0.5 < dist:
                        
                        dist = ((node_x - centroid_x)**2 + (node_y - centroid_y)**2)**0.5
                        closest_centr = centroid_id
                    
                node_centroid.append(closest_centr)            
                request_matrix[closest_centr] += number_of_occ[node_id]*0.2 
                
            N_des_coeff = np.array(request_matrix)
            self.N_des_coeff = N_des_coeff/np.sum(N_des_coeff)
            
            self.ExpProfit = scale_factor * self.N_des_coeff + np.random.uniform(-10.0, 10.0, len(self.N_des_coeff))
            
            #print("Exp Profit and ratio: ", self.ExpProfit, self.N_des_coeff)
                
    def save_map(self, folder_name):
        
        np.save(folder_name + '/Stations.npy', self.Stations)

        np.save(folder_name + '/N_max.npy', self.N_max)
        np.save(folder_name + '/N_des.npy', self.N_des)
        
        np.save(folder_name + '/ExpProfit.npy', self.ExpProfit)
        
        np.save(folder_name + '/alldists.npy', self.alldists)
        np.save(folder_name + '/allpaths.npy', self.allpaths)
        np.save(folder_name + '/coordinates.npy', self.coordinates) 
            
    def load_map(self, N_max, N_des, Stations, ExpProfit, map_height, map_width, alldists, allpaths, coordinates):
        
        self.M = len(Stations)
        
        self.N_des = N_des
        self.N_max = N_max
        
        self.Alldists = alldists
        self.Allpaths = allpaths
        self.Coordinates = coordinates
        
        self.stations = Stations
        
        self.ExpProfit = ExpProfit        