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
#planet class that inherits the values from the target class
#reads in data file about the planet information 
# We should make a documentation file that specifies how the data file should be structured, and include an example data file in the repo.
class Planet(): 
    def __init__(self, planet_file): 
        #reading in the file for planets
        data = pd.read_csv(planet_file, header = 0)
        no_pl = len(data) - 1
        PlanetID = data['PlanetID']
        Mplanet = data['MPlanet'].values
        Eccentricity = data['Eccentricity'].values
        Period = data['Period'].values
        
        self.no_pl = no_pl
        self.PlanetID = PlanetID
        self.Mplanet = Mplanet
        self.Eccentricity = Eccentricity
        self.Period = Period
        
        
# In[ ]:


#opens example file that will show information about system 
class Target(Planet): 
    def __init__(self, file, prev_rv_data = None): 
        #inherits planet class
        self.Planets = []
        super().__init__(Planet, self)
        
        read_file = file.readlines()
        #opens the file and reads it in 
        file = open(file, 'r')
        # opening a file this way causes python to keep it open, which is fine, but another way to do it is to use the following format:
        # with open(file, 'r') as f:
        #	for line in f.read_lines() ... 
        for line in read_file:
            #goes through each line and searches for keyword and assigns the value of the keyword to itself
            if line.split(' ')[0] == 'ID:': 
                ID = line.split(' ')[-1]
            self.ID = ID
            
            if line.split(' ')[0] == 'Mstar:': 
                Mstar = line.split(' ')[-1]
            Mstar.self == Mstar
            
        #if user has previous RV data
        if prev_rv_data != None:
            #reads in previous rv data and assigns them variables
            #these will be used to create plots
            # nice job setting this up!
            file_prev_rv = pd.read_csv(prev_rv_data, header = 0)

            prev_rv_time = file_prev_rv['time'] 
            prev_rv_bjd = file_prev_rv['bjd']
            prev_rv_errvel = file_prev_rv['errvel']
            prev_rv_mnvel = file_prev_rv['mnvel']

            self. prev_rv_time = prev_rv_time
            self.prev_rv_bjd = prev_rv_bjd
            self.prev_rv_errvel = prev_rv_errvel
            self.prev_rv_mnvel = prev_rv_mnvel
            
        def load_planets(self, planet_file, npl):
            for i in range(npl):
                #subset of file
                subset = 
                planet = Planet(subset)
                self.Planets.append(planet)


# In[ ]:


#creates the plots and the simulated data points
class RV_obs(Target): 
    def __init__(self, Target): 
        super().__init__(self, Target)
    
    #square root of G
    G_sqrt = 28.4329

    #mass of Jupiter in grams
    M_J = 1.899 * 10**30

    #mass of Sun in grams 
    M_sun = 1.989*10**33

    #year in seconds
    yr = 3.154*10**7 
    
    #generates the times that the user will be observing over
    #user must enter in the time they are planning on starting to observe and the date they want to end 
    #they also have to add the number of observations to establish the cadence 
    #for now we are assuming an equal cadence rather than a random one 
    def Generate_times(self, t_obs, t_end, No_obs): 
    	# For each function, we'll want to have a docstring which explains what inputs are required/what type of inputs are needed e.g. does t_obs have to be an array? Here are some examples: https://peps.python.org/pep-0257/ or you can look at the predictrvs.py file I shared
        times = np.linspace(t_obs, t_end, No_obs)
        self.times = times
        return self.times
    
    #K value is the semi amplitude of the sin curve that will be generated from the RV data
    def K_value(self, times): 
        #takes from the planet class to calculate the K value
        K = G_sqrt / (np.sqrt(1 - self.Planet.Eccentricity**2)) * self.Planet.Mplanet / M_J * (self.Planet.Mstar / M_sun)**(-2/3) * \
        (self.Planet.Period / yr)**(-1/3)
        self.K = K
        return self.K 
    
    #creates the RV values that will be put into the RV plot
    def sim_RVs(self): 
        times = self.Generate_times(t_obs, t_end, No_obs)
        Star = self.Target
        RVs = []
        #this is to make sure that we get multiple RV data points for multiple planets for more than one planet
        #loops through the number of planets and goes into the Planet class to get the data 
        for planet in self.Planet.PlanetID: 
            K = K_value(times, Planet, Target)
            #takes from the planet class and the K_value class to calculate the RV data point
            RV_pl = -K*np.sin(2*np.pi*(times - t_ref)/self.Planet.Period) + noise*np.random.randn(len(times))
            #adds RVs for every planet together 
            RV_values += RV_pl
            #all RVs is the final RV
            self.RV_values = RV_values
        return self.RV_values
    
    #plots the RV data created from the previous function
    # This looks good! My only comment is that you can move some of the code that is common to each mode outside of the if/elif statements
    def RV_plot(self, mode = 'sim'): 
        #simulated data is plotted
        if mode == 'sim': 
            fig = plt.figure()
            plt.scatter(self.times, self.RV_values)
            plt.xlabel('Time (JD)')
            plt.ylabel('RV')
            plt.show()
        #previous data loaded in from Target class is plotted
        elif mode == 'prev': 
            fig = plt.figure()
            plt.scatter(prev_rv_time.self, self.prev_rv_mnvel)
            plt.xlabel('Time (JD)')
            plt.ylabel('RV')
            plt.show()
        #both simulated data and previous data are plotted
        elif mode == 'both': 
            fig = plt.figure()
            plt.scatter(prev_rv_time.self, self.prev_rv_mnvel)
            plt.scatter(self.times, self.RV_values)
            plt.scatter()
            plt.xlabel('Time (JD)')
            plt.ylabel('RV')
            plt.show()
        return fig

class radvel_fit(RV_obs):
    def __init__(self, RV_obs): 
        super().__init__(self, RV_obs)
    
    def __repr(self)
        return 'radvel fit class'
    
    def parameters(): 
        params = no_pl
        #need a parameter for each planet so I am looping through how many planets there are and then 
        #adding those values to the parameter values 
        #defining the basis for the parameter values 
        params = radvel.Parameters(2,basis='per tc e w k')
        for i in range(len(no_pl)): 
            time_base = (times.min() + times.max())/2
            params[f'per{i}'] = radvel.Parameter(value = )
            params[f'tc{i}'] = radvel.Parameter(value = )
            params[f'secosw{i}'] = radvel.Parameter(value = )
            params[f'sesinw{i}'] = radvel.Parameter(value = )
            params[f'logk{i}'] = radvel.Parameter(value = )
            
            mod = radvel.RVModel(params, time_base=time_base)
            mod.params['dvdt'] = radvel.Parameter(value= -0.02)
            mod.params['curv'] = radvel.Parameter(value= 0.01)
            
        '''
        the rest of the code below is from the intro to astro github tutorial 
        ''' 
        mod = initialize_model() # initialize radvel.RVModel object
        like = radvel.likelihood.RVLikelihood(mod, data.t, data.vel, data.errvel, '_HIRES')
        like.params['gamma_HIRES'] = radvel.Parameter(value=0.1)
        like.params['jit_HIRES'] = radvel.Parameter(value=1.0)
        like.params['secosw1'].vary = False # set as false because we are assuming circular orbit
        like.params['sesinw1'].vary = False 
        like.params['secosw2'].vary = False # set as false because we are assuming circular orbit
        like.params['sesinw2'].vary = False 
        
        post = radvel.posterior.Posterior(like) # initialize radvel.Posterior object

        res  = optimize.minimize(
            post.neglogprob_array,     # objective function is negative log likelihood
            post.get_vary_params(),    # initial variable parameters
            method='Powell',           # Nelder-Mead also works
            )
