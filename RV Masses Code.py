#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from csv import writer 
import math
from scipy.interpolate import interp1d
import numpy as np
import os


# In[ ]:


file =  open('Example_file.txt', 'r')
#finding all of the values from the file
class Target: 
    def __init__(ID, Mstar, Mplanet, Period, Eccentricity, Prev_rvs, File_with_prev_rvs, No_obs, t_ref, t_obs, \
                t_end): 
        read_file = file.readlines()
        file = open('Example_file.txt', 'r')
        for line in read_file:
            if line.split(' ')[0] == 'ID:': 
                ID = int(stringName[len('ID:'):])
                
            if line.split(' ')[0] == 'Mstar:': 
                Mstar = int(stringName[len("Mstar:"):])
                
            if line.split(' ')[0] == 'Mplanet:': 
                Mplanet = int(stringName[len("Mplanet:"):])
                
            if line.split(' ')[0] == 'period:': 
                Period = int(stringName[len("period:"): ])
                
            if line.split(' ')[0] == 'eccentricity:': 
                Eccentricity = int(stringName[len("eccentricity:"):])
                
            if line.split(' ')[0] == 'No_obs:': 
                No_obs = int(stringName[len("No_obs:"):])
            
            if line.split(' ')[0] == 'prev_rvs:': 
                Prev_rvs = int(stringName[len('prev_rvs:'):])
            
            if line.split(' ')[0] == 'file_with_prev_rvs:': 
                File_with_prev_rvs = int(stringName[len('file_with_prev_rvs:'):])
            
            if line.split(' ')[0] == 't_ref:': 
                t_ref = int(stringName[len('t_ref:'):])
                
            if line.split(' ')[0] == 't_obs:': 
                t_obs = int(stringName[len('t_obs:'):])
                
            if line.split(' ')[0] == 't_end:': 
                t_end = int(stringName[len('t_end:')])
                
                
        return ID


# In[ ]:


class RV_obs: 
    
    #square root of G
    G_sqrt = 28.4329

    #mass of Jupiter in grams
    M_J = 1.899 * 10**30

    #mass of Sun in grams 
    M_sun = 1.989*10**33

    #year in seconds
    yr = 3.154*10**7 

    #creating times that we will use
    def Generate_times(t_obs, t_end, num = No_obs): 
        times = np.linspace(t_obs, t_end, no_obs)
        return times
    #finding the K value    
    def K_value(times, Mstar, Mplanet, Period, Eccentricity): 
        K = G_sqrt / (np.sqrt(1 - Eccentricity**2)) * Mplanet / M_J * (Mstar / M_sun)**(-2/3) * (Period / yr)**(-1/3)
        return K 
    #Generating RV values    
    def RV_values(K,times, t_ref, Period ): 
        RV_values = -K*np.sin(2*np.pi*(times - t_ref)/Period)
        return RV_values
    #generating RV vs time plot    
    def RV_plot(times, RV_values): 
        plt.figure()
        plt.scatter(times, RV_values)
        plt.xlabel('Time (JD)')
        plt.ylabel('RV')
        plt.show()

