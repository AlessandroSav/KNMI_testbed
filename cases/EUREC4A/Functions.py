#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 17:10:51 2019

@author: alessandrosavazzi
"""

#%% FUNCTIONS
#

#%% Libraries 
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import netCDF4
import time 
import os
import math
from math import pi,sin, cos, sqrt, atan2, radians
from datetime import datetime, timedelta
from netCDF4 import Dataset
import xarray as xr
#%% LIN_FIT
#
def line(x,A,B):
    return A*x + B

#%% TIC - TOC functions 
# 
def tic():
    #Homemade version of matlab tic and toc functions
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

#
def toc1():
    if 'startTime_for_tictoc' in globals():
        tot_time = time.time() - startTime_for_tictoc
        seconds_in_day = 60 * 60 * 24
        seconds_in_hour = 60 * 60
        seconds_in_minute = 60
        
        days = tot_time // seconds_in_day
        hours = (tot_time - (days * seconds_in_day)) // seconds_in_hour
        minutes = (tot_time - (days * seconds_in_day) - \
                   (hours * seconds_in_hour)) // seconds_in_minute
        seconds= tot_time - (days * seconds_in_day) - \
                  (hours * seconds_in_hour) - (minutes * seconds_in_minute)
        print ( str(days)+' days, '+str(hours)+' hours, '\
               +str(minutes)+' minutes, '+str(np.round(seconds,3))+' seconds.')
    else:
        print ("Toc: start time not set")

#%% POINTS_between coordinates
def points_along_axis(edge1, edge2, n=20):
    return np.column_stack((np.linspace(edge1[0], edge2[0], n+1),
               np.linspace(edge1[1], edge2[1], n+1)))

#%%
# find the closest element to 'node' within 'nodes' 
def closest_node(node, nodes):
    # nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1) 
    dist = my_distance_geo(nodes.iloc[np.argmin(dist_2)]['lat'],\
                           nodes.iloc[np.argmin(dist_2)]['lon'],\
                               node[1],node[0])
    return  nodes.index[np.argmin(dist_2)], abs(dist) 

#%%
def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))
#%%  Distance on globe in km 
def my_distance_geo(lat1,lon1,lat2,lon2):
    # approximate radius of earth in km
    R = 6373.0
    
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    
    dlon = lon1 - lon2
    dlat = lat1 - lat2
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance = R * c
    if (dlon<0) or (dlat<0):
        distance = - distance 
    return distance