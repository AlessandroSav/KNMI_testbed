#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 13:34:40 2020

EXTRACT DAYS FROM HARMONIE OUTPUTS

@author: alessandrosavazzi
"""
import numpy as np
import pandas as pd
import tarfile
import glob, os
import xarray as xr
import matplotlib.pyplot as plt
from math import radians
####                        IMPORT MYFUNCTIONS                            ####
from Functions import my_distance_geo, closest_node

#%%
##############################################################################
### Parameters to set 
str_time  = 2020020100  #yyyymmddhh
end_time  = 2020020200
##############################################################################
read_dir= '/Users/alessandrosavazzi/Desktop/WORK/PhD_Year1/DALES/DALES/Cases/EUREC4A/'
# read_dir  = os.path.abspath('{}/../EURECA')
# write_dir = os.path.abspath('{}/../extracted/')
# print ('read_dir: '+read_dir)
# print ('write_dir: '+write_dir)

os.chdir(read_dir)
for file in glob.glob("*.tar"):
    print('extracting '+file)
    tar = tarfile.open(file)
    for member in tar.getnames():
        if (int(member[-17:-7]) <= end_time) and \
           (int(member[-17:-7]) >= str_time):
            tar.extract(member,path='./extracted/')
#%%
print('End.')

#%%
import xarray as xr            
temp = xr.open_dataset('/Users/alessandrosavazzi/Desktop/WORK/PhD_Year1/DALES/DALES/Cases/cabaw/HARMONIE/LES_forcing_2016053018.nc') 




