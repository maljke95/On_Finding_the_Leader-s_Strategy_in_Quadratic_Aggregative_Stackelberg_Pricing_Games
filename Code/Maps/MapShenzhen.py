# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 22:41:51 2022

@author: marko
"""
import os 
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

from Maps.Map_template import Map_template

class MapShenzhen(Map_template):
    
    def __init__(self, batch_id=2, arrivals_id=0, path_data=None, path_arrivals=None, path_coordinates=None):
        
        super(MapShenzhen, self).__init__()
        
        self.generate_map(batch_id, arrivals_id, path_data, path_arrivals, path_coordinates)

    def generate_map(self, batch_id=2, arrivals_id=0, path_data=None, path_arrivals=None, path_coordinates=None):                              # can be any number from 1 to 10
               
        if path_data is None:
            path_data = os.getcwd() + '/Data/input_data.mat'
        mat  = scipy.io.loadmat(path_data)
        
        if path_arrivals is None:
            path_data = os.getcwd() + '/Data/arrival batchs/arrivals_40k80k.mat'
        mat1 = scipy.io.loadmat(path_arrivals)
            
        if path_coordinates is None:
            path_coordinates = os.getcwd() + '/Data/Grid/coordinates.mat'
        mat2 = scipy.io.loadmat(path_coordinates)
        
        arrivals = mat1['arrival_batch'][arrivals_id]
        
        alldists = mat['alldists']
        allpaths = mat['allpaths']
        mdetour = float(mat['mdetour'])
        tstep = float(mat['tstep'])
        wtol = float(mat['wtol'])
    
        coordinates = mat2['coordinates']
        coordinates[:,2] = coordinates[:,2]-1                                       # fist and second column are node coordinates and the third one is the node label
            
        temp_arrtime = (arrivals[batch_id][0][0][0]).flatten()
        temp_orig    = (arrivals[batch_id][0][0][1] -1).flatten()
        temp_dest    = (arrivals[batch_id][0][0][2] - 1).flatten()
        temp_trip    = (arrivals[batch_id][0][0][3]).flatten()
        
        self.arrtime = []
        self.orig    = []
        self.dest    = []
        self.trip    = []
        
        for idx in range(len(temp_arrtime)):           
            if not (temp_orig[idx] == temp_dest[idx]):
                
                self.arrtime.append(temp_arrtime[idx])
                self.orig.append(temp_orig[idx])
                self.dest.append(temp_dest[idx])
                self.trip.append(temp_trip[idx])

        self.arrtime = np.array(self.arrtime)
        self.orig    = np.array(self.orig)
        self.dest    = np.array(self.dest)
        self.trip    = np.array(self.trip)
                
        self.calculate_origin_destination()
        
        allpaths = allpaths-1   
        
        allintersect = []
        N_nodes = allpaths.shape[1]
        
        for curr_intersection in range(N_nodes):

            list_of_adjacent_intersections = []
        
            for idx in range(N_nodes):
                if allpaths[curr_intersection, idx] == curr_intersection and not(curr_intersection == idx):
                    list_of_adjacent_intersections.append(idx)
                    
            allintersect.append(list_of_adjacent_intersections)
            
        self.allpaths = allpaths
        self.alldists = alldists
        self.coordinates = coordinates
        
        self.allintersect = allintersect
        
        self.wtol = wtol
        self.tstep = tstep
        self.mdetour = mdetour
        
        #----- Values of the next request arrivals in the system -----
        
        self.next_arrtime = np.copy(self.arrtime)
        self.next_orig    = np.copy(self.orig)
        self.next_dest    = np.copy(self.dest)
        self.next_trip    = np.copy(self.trip)
            
    def vel(self, nVeh):
    
        """ Compute network speed based on accumulation """
        
        m = nVeh/1000
        tveh = 60
        
        if m<=0.6*tveh:
            velocity = 30.8*np.exp(-m*0.145*20/tveh)
        elif m<=tveh:
            velocity = 5.4-(m-0.6*tveh)*0.71*20/tveh;
        else:
            velocity = 0
            
        if velocity<0:
            velocity = 0
            
        velocity = 36/30.8*velocity
        
        return velocity 
    
    def plot_MFD(self, save_fig=False, folder_name=None):
        
        fig = plt.figure(dpi=180)
        ax  = fig.add_subplot()
        
        accum = np.linspace(0.0, 60000.0, 10000)
        vec_v = []
        for acc in accum:
            vec_v.append(self.vel(acc))
        
        ax.plot(accum/1000.0, vec_v)
        ax.set_xlabel('Accumulation')
        ax.set_ylabel('Space mean speed')
        ax.grid('on')     
        
        if save_fig:
            
            tikzplotlib.clean_figure()
            tikzplotlib.save(folder_name + "/MFD.tex")
            fig.savefig(folder_name + "/MFD.jpg", dpi=180)        
    
#if __name__ == '__main__':
    
    #m = MapShenzhen()
    #m.plot_map(show_plot=True, plot_orig=True, plot_dest=False, save_fig=False, folder_name=None)