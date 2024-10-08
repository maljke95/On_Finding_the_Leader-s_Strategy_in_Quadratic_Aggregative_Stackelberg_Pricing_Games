# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 22:35:44 2022

@author: marko
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import tikzplotlib

class Map_template(object):
    
    def __init__(self):
        
        self.alldists = None
        self.allpaths = None
        self.coordinates = None
        self.allintersect = None

        self.arrtime = None
        self.orig    = None
        self.dest    = None
        self.trip    = None
        
        #----- Set of arrays for remaining requests as the time goes by -----
        
        self.next_arrtime = None
        self.next_orig = None
        self.next_dest = None
        self.next_trip = None
        
        self.wtol = None
        self.tstep = None
        self.mdetour = None
        
        #----- Some information oabout origin-destination distribution -----
        
        self.number_of_occ_origin = None
        self.number_of_occ_dest = None
        
        self.current_request_id = 0

    def generate_map(self):
    
        pass
    
    def save_map(self, folder_name):
        
        np.save(folder_name + '/alldists.npy', self.alldists)
        np.save(folder_name + '/allpaths.npy', self.allpaths)
        np.save(folder_name + '/coordinates.npy', self.coordinates)
        
    def fetch_requests(self, t_start, t_end):
        
        list_of_arrtime = []
        list_of_origins = []
        list_of_dest    = []
        list_of_trip    = []
        list_of_ids     = []
        
        for arr_id in range(len(self.next_arrtime)):
            
            arr = self.next_arrtime[arr_id]
            
            if arr >= t_end:
                
                self.next_arrtime = self.next_arrtime[arr_id:]
                self.next_dest    = self.next_dest[arr_id:]
                self.next_orig    = self.next_orig[arr_id:]
                self.next_trip    = self.next_trip[arr_id:]
                
                return list_of_arrtime, list_of_origins, list_of_dest, list_of_trip ,list_of_ids
            
            else:
                
                list_of_arrtime.append(arr)
                list_of_origins.append(self.next_orig[arr_id])
                list_of_dest.append(self.next_dest[arr_id])
                list_of_trip.append(self.next_trip[arr_id])               
                list_of_ids.append(self.current_request_id)
                
                self.current_request_id += 1
                
    def sample_private_and_ride_hailing_requests(self, list_of_arrtime, list_of_origins, list_of_dest, list_of_trip, list_of_ids,  prob=0.2):
        
        list_of_private_arrtime = []
        list_of_private_orig = []
        list_of_private_dest = []
        list_of_private_trip = []
        list_of_private_ids  = []

        list_of_ridehailing_arrtime = []
        list_of_ridehailing_orig = []
        list_of_ridehailing_dest = []
        list_of_ridehailing_trip = []
        list_of_ridehailing_ids  = []
                
        for idx in range(len(list_of_arrtime)):
            r = np.random.rand()
            if r >= prob:
                
                list_of_private_arrtime.append(list_of_arrtime[idx])
                list_of_private_orig.append(list_of_origins[idx])
                list_of_private_dest.append(list_of_dest[idx])
                list_of_private_trip.append(list_of_trip[idx])
                list_of_private_ids.append(list_of_ids[idx])
                
            else:
                
                list_of_ridehailing_arrtime.append(list_of_arrtime[idx])
                list_of_ridehailing_orig.append(list_of_origins[idx])
                list_of_ridehailing_dest.append(list_of_dest[idx])
                list_of_ridehailing_trip.append(list_of_trip[idx])  
                list_of_ridehailing_ids.append(list_of_ids[idx])
                
        return list_of_private_arrtime, list_of_private_orig, list_of_private_dest, list_of_private_trip, list_of_private_ids,\
            list_of_ridehailing_arrtime, list_of_ridehailing_orig, list_of_ridehailing_dest, list_of_ridehailing_trip, list_of_ridehailing_ids
            
    def calculate_origin_destination(self):
        
        max_node_id_orig = np.max(self.orig)
        max_node_id_dest = np.max(self.dest)

        number_of_occ_origin = np.zeros(max_node_id_orig + 1)
        number_of_occ_dest   = np.zeros(max_node_id_dest + 1)

        for elem_id in range(len(self.orig)):
    
            number_of_occ_origin[self.orig[elem_id]] += 1
            number_of_occ_dest  [self.dest[elem_id]] += 1
            
        self.number_of_occ_origin = number_of_occ_origin
        self.number_of_occ_dest   = number_of_occ_dest
                
    
    def vel(self, nVeh):
        
        pass
    
    def plot_MFD(self, save_fig=False, folder_name=None):
        
        pass
    
    def plot_map(self, show_plot=False, plot_orig=False, plot_dest=False, save_fig=False, folder_name=None):

        if plot_orig:
            
            fig_origin, ax_origin = plt.subplots(dpi=180)
            
            nodes_origin = len(self.number_of_occ_origin)
            norm_origin = plt.Normalize(self.number_of_occ_origin.min(), self.number_of_occ_origin.max())
            
            ax_origin.scatter(self.coordinates[:nodes_origin,0], self.coordinates[:nodes_origin,1], c=self.number_of_occ_origin, norm=norm_origin)
            cbar_origin = fig_origin.colorbar(cm.ScalarMappable(norm = norm_origin), ax=ax_origin)
            cbar_origin.ax.set_ylabel('Requests')
            
            ax_origin.grid('on')
            ax_origin.xaxis.set_visible(False)
            ax_origin.yaxis.set_visible(False)

            if save_fig:
                
                fig_origin.savefig(folder_name + "/Map_origins.jpg", dpi=180)              
                tikzplotlib.save(folder_name + "/Map_origins.tex")
            
        if plot_dest:
            
            fig_dest, ax_dest = plt.subplots(dpi=180)
            
            nodes_dest = len(self.number_of_occ_dest)
            norm_dest = plt.Normalize(vmin=np.min(self.number_of_occ_dest), vmax=np.max(self.number_of_occ_dest))
            
            ax_dest.scatter(self.coordinates[:nodes_dest,0], self.coordinates[:nodes_dest,1], c=self.number_of_occ_dest, norm=norm_dest)            
            cbar_dest = fig_dest.colorbar(cm.ScalarMappable(norm = norm_dest), ax=ax_dest)
            cbar_dest.ax.set_ylabel('Requests')
            

            ax_dest.grid('on')
            ax_dest.xaxis.set_visible(False)
            ax_dest.yaxis.set_visible(False)
            
            if save_fig:
            
                fig_dest.savefig(folder_name + "/Map_destination.jpg", dpi=180)
                tikzplotlib.save(folder_name + "/Map_destination.tex") 
        
        if show_plot:
            plt.show()
            
        plt.close()
        
    def voronoi_regions(self, list_of_centroids, partition_destinations=False, show_plot=False, save_fig=False, folder_name=False):
        
        list_of_colors = ['tab:red', 'tab:orange', 'tab:gray', 'tab:cyan', 'tab:brown', 'tab:pink', 'tab:green', 'tab:olive', 'tab:purple']
        
        number_of_nodes = len(self.number_of_occ_origin)
        number_of_occ = self.number_of_occ_origin
        
        if partition_destinations:
            number_of_nodes = len(self.number_of_occ_dest)
            number_of_occ = self.number_of_occ_dest
            
        N_centr = len(list_of_centroids)
        
        request_matrix = np.zeros(N_centr)
        node_centroid = []
        
        for node_id in range(number_of_nodes):
            
            node_x = self.coordinates[node_id, 0]
            node_y = self.coordinates[node_id, 1]
            
            closest_centr = None
            dist = None
            
            for centroid_id in range(N_centr):
                
                centroid = list_of_centroids[centroid_id]
                
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
        
        list_of_freq = []
        for node_id in range(number_of_nodes):
            
            centroid_id = node_centroid[node_id]
            list_of_freq.append(request_matrix[centroid_id])
            
        list_of_freq = np.array(list_of_freq)
        
        fig, ax = plt.subplots(dpi=180)
        
        norm_occ = plt.Normalize(request_matrix.min(), request_matrix.max())        
        ax.scatter(self.coordinates[:number_of_nodes,0], self.coordinates[:number_of_nodes,1], c=list_of_freq, norm=norm_occ)

        for i in range(len(list_of_centroids)):
                
            station = list_of_centroids[i]
            ax.plot(self.coordinates[station,0], self.coordinates[station,1], marker='s', markersize = 10, color = list_of_colors[i], label = 'M '+str(i+1))

        cbar_occ = fig.colorbar(cm.ScalarMappable(norm = norm_occ), ax=ax)
            
        ax.legend()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)  
        
        if save_fig:
            
            fig.savefig(folder_name + "/Arrivals_matrix.jpg", dpi=180)

            tikzplotlib.save(folder_name + "/Arrivals_matrix.tex")
        
        if show_plot:
            plt.show()
            
        plt.close()        
        
        
        