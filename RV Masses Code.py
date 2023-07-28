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

from scipy import optimize
import corner
import radvel
import radvel.likelihood
from radvel.plot import orbit_plots, mcmc_plots


# In[269]:


#planet class that inherits the values from the target class
#reads in data file about the planet information 
class Planet(): 
    def __init__(self, PlanetID, Period, Eccentricity, Mplanet, Tc): 
        
        self.PlanetID = PlanetID
        self.Mplanet = Mplanet
        self.Eccentricity = Eccentricity
        self.Period = Period
        self.Tc = Tc
        
    #defining what the name of the class is when an object created is called   
    def __repr__(self):
        return 'Planet Class'


# In[270]:


#Target class loads in system data, gets attributes for multiple planets, and gets data about previous rv data
class Target(): 
    #needs a file for the target info and optional previous RV data
    #user will have to enter in telescope 
    def __init__(self, file, telescope, prev_rv_data = None): 
        #inherits planet class and creates empty array so that planet info will be stored
        self.Planets = []
        
        self.file = file
        
        #opens the file and reads it in 
        target_file = pd.read_csv(self.file, header = 0)
        ID = target_file['ID'][0]
        Mstar = target_file['Mstar'][0]
        
        self.ID = ID
        self.Mstar = Mstar
        self.prev_rv_data = None
        self.no_pl = 0
        self.telescope = telescope
        #if user has previous RV data
        if prev_rv_data != None:
            #reads in previous rv data and assigns them variables
            #these will be used to create plots
            file_prev_rv = pd.read_csv(prev_rv_data, header = 0)
            self.prev_rv_data = file_prev_rv

            prev_rv_time = file_prev_rv['time'] 
            prev_rv_bjd = file_prev_rv['bjd']
            prev_rv_errvel = file_prev_rv['errvel']
            prev_rv_mnvel = file_prev_rv['mnvel']
            prev_telescope = file_prev_rv['tel']

            self.prev_rv_time = prev_rv_time
            self.prev_rv_bjd = prev_rv_bjd
            self.prev_rv_errvel = prev_rv_errvel
            self.prev_rv_mnvel = prev_rv_mnvel
            self.prev_telescope = prev_telescope
            
    #defining what the name of the class is when an object created is called               
    def __repr__(self):
        return 'Target Class:'#, self.Target.ID

    #creating a function that creates multiple objects of the planet class to create multiple planets
    #this will be read in by the csv file provided into the planet class
    def load_planets(self, planet_file):

        #reading in the planet file and assigning it variable data 
        data_pl = pd.read_csv(planet_file, header = 0)
        
        no_pl = len(data_pl)
        self.no_pl = no_pl

        for i in range(self.no_pl):
            #subset of file will be the ith row that is iterated 
            subset = data_pl.iloc[i]
            
            #getting the values from the file for each planet in the system 
            planetID = subset['PlanetID']
            mplanet = subset['MPlanet']
            eccentricity = subset['Eccentricity']
            period = subset['Period']
            #tc = time of transit
            tc = subset['tc']

            #creating individual objects for each planet and adding them to the list as defined in __init__
            planet = Planet(planetID, period, eccentricity, mplanet, tc)
            self.Planets.append(planet)


# In[127]:


#square root of G
G_sqrt = 28.4329

#mass of Jupiter in grams
M_J = 1.899 * 10**30

#mass of Sun in grams 
M_sun = 1.989*10**33

#year in seconds
yr = 3.154*10**7 


# In[271]:


#creates the plots and the simulated data points
class RV_obs(Target): 
    def __init__(self, Target, t_obs, t_end, No_obs, t_ref): 
        self.Target = Target
        self.t_obs = t_obs
        self.t_end = t_end
        self.No_obs = No_obs
        self.t_ref = t_ref
    
    
    #generates the times that the user will be observing over
    #user must enter in the time they are planning on starting to observe and the date they want to end 
    #they also have to add the number of observations to establish the cadence 
    #for now we are assuming an equal cadence rather than a random one 
    def Generate_times(self): 
        times = np.linspace(self.t_obs, self.t_end, self.No_obs)
        self.times = times
        return self.times
    
    #K value is the semi amplitude of the sin curve that will be generated from the RV data
    def K_value(self, times, planet, star): 
        #takes from the planet class to calculate the K value
        K = G_sqrt / (np.sqrt(1 - planet.Eccentricity**2)) * planet.Mplanet / M_J * \
        (star.Mstar / M_sun)**(-2/3) * (planet.Period / yr)**(-1/3)
        self.K = K
        return self.K 
    
    #creates the RV values that will be put into the RV plot
    def sim_RVs(self, noise = 0): 
        times = self.Generate_times()
        Star = self.Target
        RVs = []
        RV_values = 0
        #come back to this later when we can create errors
        self.RV_errs = np.array([1]*len(times))
        #this is to make sure that we get multiple RV data points for multiple planets for more than one planet
        #loops through the number of planets and goes into the Planet class to get the data 
        for planet in Star.Planets: 
            K = self.K_value(times, planet, Star)
            
            #takes from the planet class and the K_value class to calculate the RV data point
            RV_pl = -K*np.sin(2*np.pi*(times - self.t_ref)/planet.Period) + noise*np.random.randn(len(times))
            #adds RVs for every planet together 
            RV_values += RV_pl
            #all RVs is the final RV
            self.RV_values = RV_values
        return self.RV_values
    
    #plots the RV data created from the previous function
    def RV_plot(self, mode = 'sim'): 
        #simulated data is plotted
        if mode == 'sim': 
            fig = plt.figure()
            plt.scatter(self.times, self.RV_values)
            plt.xlabel('Time (JD)')
            plt.ylabel('RV')
            plt.title("Simulated Data")
            self.fig = fig
            return self.fig
        #previous data loaded in from Target class is plotted
        elif mode == 'prev': 
            fig = plt.figure()
            plt.scatter(prev_rv_time.self, self.prev_rv_mnvel)
            plt.xlabel('Time (JD)')
            plt.ylabel('RV')
            plt.title("Previous RV Data")
            self.fig = fig
            return self.fig
        #both simulated data and previous data are plotted
        elif mode == 'both': 
            fig = plt.figure()
            plt.scatter(prev_rv_time.self, self.prev_rv_mnvel, label = 'previous RV')
            plt.scatter(self.times, self.RV_values, label = 'simulated')
            plt.scatter()
            plt.xlabel('Time (JD)')
            plt.ylabel('RV')
            plt.legend()
            self.fig = fig
            return self.fig
        
    #creating function for the parameters
    #parameters to vary is by default an empty list, if user wants to input parameters to vary then they will have to 
    #define it when they call this function 
    def parameters(self, params_to_vary = []): 
        #need a parameter for each planet so I am looping through how many planets there are and then 
        #adding those values to the parameter values 
        #defining the basis for the parameter values 
        Star = self.Target
        no_pl = self.Target.no_pl
        
        params = radvel.Parameters(2,basis='per tc e w k')
        for i in range(no_pl): 
            time_base = (self.times.min() + self.times.max())/2
            params[f'per{i}'] = radvel.Parameter(value = Star.Planets[i].Period)
            if Star.Planets.Tc == np.nan: 
                tc = np.random.uniform(self.times.min(), self.times.max())
                params[f'tc{i}'] = radvel.Parameter(value = tc)
            else: 
                params[f'tc{i}'] = radvel.Parameter(value = Star.Planets[i].tc)
            params[f'e{i}'] = radvel.Parameter(value = Star.Planets[i].Eccentricity)
            params[f'w{i}'] = radvel.Parameter(value = np.pi/2)
            params[f'k{i}'] = radvel.Parameter(value = Star.Planets[i].K)

        mod = radvel.RVModel(params, time_base=time_base)
        mod.params['dvdt'] = radvel.Parameter(value = 0.02)
        mod.params['curv'] = radvel.Parameter(value = 0.01)

        
        like = radvel.likelihood.RVLikelihood(mod, self.times, self.RV_values, self.RV_errs)
        #make a set a gamma_prevtelescope and jit_prevtelescope and a set for the simulated telescope
        #get name of telescope used in previous observations and use that to track uncertainty in telescope 
        #do the same for simulated telescope data 
        like.params[f'gamma_{self.Target.prev_telescope}'] = radvel.Parameter(value = 0.1)
        like.params[f'jit_{self.Target.prev_telescope}'] = radvel.Parameter(value = 1.0)
        like.params[f'gamma_{self.Target.telescope}'] = radvel.Parameter(value=0.1)
        like.params[f'jit_{self.Target.telescope}'] = radvel.Parameter(value=1.0)
        
        #for each of the parameters for each planet, set them to vary as true or false
        #want user to pass in a list of parameters they want to vary 
        
        #for loop is for each planet's parameters 
        for i in range(len(no_pl)): 
            like.params[f'secosw{i}'].vary = False
            like.params[f'sesinw{i}'].vary = False
            like.params[f'per{i}'].vary = False
            like.params[f'tc{i}'].vary = False
        #parameters that are for the previous telescope from previous data and the current Target telescope    
        like.params[f'gamma_{self.Target.prev_telescope}'].vary = False
        like.params[f'jit_{self.Target.prev_telescope}'].vary = False
        like.params[f'gamma_{self.Target.telescope}'].vary = False
        like.params[f'jit_{self.Target.telescope}'].vary = False
        
        #looping though array of parameters to vary and setting them equal to true 
        if len(params_to_vary) > 0:
            for i in params_to_vary:
                like.params[params_to_vary[i]].vary = True
        
        '''
        the rest of the code below is from the intro to astro github tutorial 
        ''' 
        #initializes radvel.Posterior object 
        post = radvel.posterior.Posterior(like)
        
        #maximizes likelihood
        res  = optimize.minimize(
            post.neglogprob_array,     # objective function is negative log likelihood
            post.get_vary_params(),    # initial variable parameters
            method='Nelder-Mead',           # Powell also works
            )
        
        #ready-made plots that radvel has
        matplotlib.rcParams['font.size'] = 12

        RVPlot = orbit_plots.MultipanelPlot(post)
        RVPlot.plot_multipanel()

        matplotlib.rcParams['font.size'] = 18
        return RVPlot


