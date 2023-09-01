#!/usr/bin/env python
# coding: utf-8

# In[1187]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from csv import writer 
import math
from scipy.interpolate import interp1d
import numpy as np
import os
import astropy.units as u

from scipy import optimize
import corner
import radvel
import radvel.likelihood
from radvel.plot import orbit_plots, mcmc_plots

from astropy.timeseries import LombScargle
from astropy.time import Time


# In[1063]:


#planet class that inherits the values from the target class
#reads in data file about the planet information 
class Planet(): 
    def __init__(self, PlanetID, Period, Eccentricity, Mplanet, Tc, K=None): 
        
        self.PlanetID = PlanetID
        self.Mplanet = Mplanet
        self.Eccentricity = Eccentricity
        self.Period = Period
        self.Tc = Tc
        self.K = K
        
    #defining what the name of the class is when an object created is called   
    def __repr__(self):
        return 'Planet Class'


# In[1358]:


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
            file_prev_rv = pd.read_csv(prev_rv_data, header = 1)
            self.file_prev_rv = file_prev_rv
            self.prev_rv_data = True

            prev_rv_jd = file_prev_rv['bjd']
            prev_rv_rv = file_prev_rv['mnvel']
            prev_rv_err = file_prev_rv['errvel']
            prev_telescope = file_prev_rv['tel']

            self.prev_rv_jd = prev_rv_jd
            self.prev_rv_rv = prev_rv_rv
            self.prev_rv_err = prev_rv_err
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
            eccentricity = float(subset['Eccentricity'])
            period = float(subset['Period'])
            #tc = time of transit
            tc = float(subset['tc'])

            #creating individual objects for each planet and adding them to the list as defined in __init__
            planet = Planet(planetID, period, eccentricity, mplanet, tc)
            self.Planets.append(planet)


# In[1359]:


#square root of G
G_sqrt = 28.4329

#mass of Jupiter in grams
M_J = 1.899 * 10**30

#mass of Sun in grams 
M_sun = 1.989*10**33

#year in seconds
yr = 3.154*10**7 


# In[1360]:


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
        print (planet.Eccentricity, planet.Mplanet)
        K = G_sqrt / (np.sqrt(1 - planet.Eccentricity**2)) * (planet.Mplanet*u.M_earth / u.M_earth) * \
        (star.Mstar*u.M_sun / u.M_sun)**(-2/3) * (planet.Period*u.yr / u.yr)**(-1/3)
        planet.K = K
        return planet.K 
    
    #creates the RV values that will be put into the RV plot
    def sim_RVs(self, noise = 0): 
        times = self.Generate_times()
        Star = self.Target
        RVs = []
        RV_values = 0
        #come back to this later when we can create errors
        self.RV_errs = np.array([10]*len(times))
        #this is to make sure that we get multiple RV data points for multiple planets for more than one planet
        #loops through the number of planets and goes into the Planet class to get the data 
        for planet in Star.Planets: 
            K = self.K_value(times, planet, Star)
            print(K)
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
            plt.ylabel('RV (m/s)')
            plt.title("Simulated Data")
            self.fig = fig
            return self.fig
        #previous data loaded in from Target class is plotted
        elif mode == 'prev': 
            fig = plt.figure()
            plt.scatter(self.Target.prev_rv_jd, self.Target.prev_rv_rv)
            plt.xlabel('Time (JD)')
            plt.ylabel('RV (m/s)')
            plt.title("Previous RV Data")
            self.fig = fig
            return self.fig
        #both simulated data and previous data are plotted
        elif mode == 'both': 
            fig = plt.figure()
            plt.scatter(self.Target.prev_rv_jd, self.Target.prev_rv_rv, label = 'previous RV')
            plt.scatter(self.times, self.RV_values, label = 'simulated')
            plt.xlabel('Time (JD)')
            plt.ylabel('RV (m/s)')
            plt.legend()
            self.fig = fig
            return self.fig
        else: 
            print("Please input a valid mode.")
    #creating function for the parameters
    #parameters to vary is by default an empty list, if user wants to input parameters to vary then they will have to 
    #define it when they call this function 
    def parameters(self, params_to_vary = []): 
        #need a parameter for each planet so I am looping through how many planets there are and then 
        #adding those values to the parameter values 
        #defining the basis for the parameter values 
        Star = self.Target
        no_pl = self.Target.no_pl
        instnames = np.array(Star.telescope)
        
        params = radvel.Parameters(no_pl, basis='per tp e w k')
        self.params = params
        for i in range(1, no_pl+1): 
            time_base = (self.times.min() + self.times.max())/2
            params[f'per{i}'] = radvel.Parameter(value = Star.Planets[i-1].Period)
            params[f'tp{i}'] = radvel.Parameter(value = Star.Planets[i-1].Tc)
            params[f'e{i}'] = radvel.Parameter(value = Star.Planets[i-1].Eccentricity)
            params[f'w{i}'] = radvel.Parameter(value = np.pi/2)
            params[f'k{i}'] = radvel.Parameter(value = Star.Planets[i-1].K)
        
        
#         params2 = params.basis.to_any_basis(params,'per tc secosw sesinw k')
        mod = radvel.RVModel(params, time_base=time_base)
        print (mod.params)
        mod.params['dvdt'] = radvel.Parameter(value = -0.02)
        mod.params['curv'] = radvel.Parameter(value = 0.01)
        
        like = radvel.likelihood.RVLikelihood(mod, self.times, self.RV_values, self.RV_errs, suffix="_"+Star.telescope)
        #make a set a gamma_prevtelescope and jit_prevtelescope and a set for the simulated telescope
        #get name of telescope used in previous observations and use that to track uncertainty in telescope 
        #do the same for simulated telescope data 
        if self.Target.prev_rv_data != None: 
            like.params[f'gamma_{self.Target.prev_telescope}'] = radvel.Parameter(value = 0.1)
            like.params[f'jit_{self.Target.prev_telescope}'] = radvel.Parameter(value = 1.0)
        like.params[f'gamma_{self.Target.telescope}'] = radvel.Parameter(value=0.1, vary=True)
        like.params[f'jit_{self.Target.telescope}'] = radvel.Parameter(value=1.0, vary=True)
        like = radvel.likelihood.CompositeLikelihood([like])
        
        #for each of the parameters for each planet, set them to vary as true or false
        #want user to pass in a list of parameters they want to vary 
        
        #for loop is for each planet's parameters 
        for i in range(1, no_pl+1): 
            #like.params[f'secosw{i}'].vary = False
            #like.params[f'sesinw{i}'].vary = False
            like.params[f'e{i}'].vary = True
            like.params[f'per{i}'].vary = True
            like.params[f'tp{i}'].vary = True
            like.params[f'k{i}'].vary = True
            like.params[f'w{i}'].vary = False
#         #parameters that are for the previous telescope from previous data and the current Target telescope
        if self.Target.prev_rv_data != None:
            like.params[f'gamma_{self.Target.prev_telescope}'].vary = False
            like.params[f'jit_{self.Target.prev_telescope}'].vary = False
        like.params[f'gamma_{self.Target.telescope}'].vary = False
        like.params[f'jit_{self.Target.telescope}'].vary = False
                
        #looping though array of parameters to vary and setting them equal to true 
        if len(params_to_vary) > 0:
            for i in params_to_vary:
                like.params[params_to_vary[i]].vary = True
            
        '''
        the rest of the code below is from the intro to astro github tutorial/radvel K2-24 tutorial  
        ''' 
        #initializes radvel.Posterior object 
        post = radvel.posterior.Posterior(like)

        # add priors
        for i in range(1, no_pl+1):
            post.priors += [radvel.prior.Gaussian(f'tp{i}', Star.Planets[i-1].Tc, 0.1)]
            post.priors += [radvel.prior.Gaussian(f'per{i}', Star.Planets[i-1].Period, 0.5)]
        post.priors += [radvel.prior.HardBounds(f'jit_{self.Target.telescope}', -20.0, 20.0)]
        post.priors += [radvel.prior.EccentricityPrior(no_pl)]
        
#         print("Initial loglikelihood = %f" % post.logprob())
        print (post.name_vary_params())
    
        # Perform Max-likelihood fitting
        res  = optimize.minimize(
            post.neglogprob_array,
            post.get_vary_params(),
            method='Powell',
            options=dict(maxiter=100000,maxfev=100000,xtol=1e-8)
            )
        print("Final loglikelihood = %f" % post.logprob())
        print (post.name_vary_params())
        print (post)
        
        #ready-made plots that radvel has, from intro to astro tutorial on github
        RVPlot = orbit_plots.MultipanelPlot(post)
        RVPlot.plot_multipanel()
        return RVPlot
    
   


# In[1365]:


class cadence(RV_obs): 
    def __init__(self, RV_obs): 
        self.RV_obs = RV_obs

    #creating lomb scargle periodigram 
    def lombscargle(self, mode = 'previous data'): 
        if mode == 'residuals': 
            frequency, power = LombScargle(self.RV_obs.times, radvel.likelihood.RVLikelihood.residuals).autopower()
        elif mode == 'previous data': 
            ls = LombScargle(t, y, dy)
            frequency, power = ls.autopower()
            frequency, power = LombScargle(self.RV_obs.Target.prev_rv_jd, self.RV_obs.Target.prev_rv_rv).autopower()
        else: 
            print("Please write a valid input for mode")

        false_alarm = ls.false_alarm_probability(power.max())
        #if false_alarm < 0.01: 
        peak = power.max()
        self.peak = peak
        #plot for periodigram 
        fig = plt.figure()
        plt.plot(frequency, power)
        plt.ylabel("Frequency")
        plt.xlabel("Lomb-Scargle Power")
        plt.show()
        return fig

    #this function will phase fold 
    def phase_fold(self, mode = 'default'): 
        #converts JD time to orbital phase, taken from radvel docs
        if mode == 'default':
            P =  self.RV_obs.Target.Planets[0].Period
            tp =  self.RV_obs.Target.Planets[0].Tc
            t = self.RV_obs.Target.prev_rv_jd
            rv = self.RV_obs.Target.prev_rv_rv
        #if the user wants to phase fold on the data from lomb scargle 
        elif mode == 'lomb scargle': 
            P =  self.peak
            tp =  self.RV_obs.Target.Planets[0].Tc
            if lombscargle.mode == 'previous data': 
                t = self.RV_obs.Target.prev_rv_jd
                rv = self.RV_obs.Target.prev_rv_rv
            elif lombscargle.mode == 'residuals': 
                t = self.RV_obs.times
                rv = radvel.likelihood.RVLikelihood.residuals
        else: 
            print("Please write a valid input for mode.")
            
        self.period = P
        self.tp = tp
        phase = np.mod(t - self.tp, self.period)
        phase /= self.period
        self.phase = phase
        #creating phase fold plot
        phase_plot = plt.figure()
        plt.scatter(phase, rv)
        plt.xlabel('Phase')
        plt.ylabel('RV (m/s)')
        plt.show()
        return self.phase, phase_plot
    
    #finding gaps in the phases 
    def phase_gaps(self):
        self.phase = np.sort(self.phase)
        distance = []
        #taking the difference between two points of the phase and adding them to an array
        for i in range(len(self.phase)): 
            #since we are subtracting from the next iteration, if the ith value is equal to the length of the 
            #phase we want to break out of the loop 
            if i == len(self.phase)-1: 
                break
            else: 
                diff = np.absolute(self.phase[i+1] - self.phase[i])
                distance = np.append(distance, diff)
        self.distance = distance
        #averaging out the distances between each of the phases and finding std
        avg_distance = np.mean(distance)
        self.avg_distance = avg_distance 
        self.std = np.std(self.distance)
        
        #loops through the distance array and flags the phase that relates to the distance 
        #then converts the phases to days within the period and adds them to an array 
        flagged_phase = [] 
        phase_obs = []
        i = 0
        j = 0 
        while i <= len(distance):
            if distance[i] > self.avg_distance: 
                #turning the flagged phases into days within the period and adding them to array phase_obs
                #phase_obs is an array of days where you should observe 
                flagged_day = self.phase[j] * self.period
                flagged_day_2 = self.phase[j+1]*self.period
                steps = int(np.round((flagged_day_2 - flagged_day) * self.period))
                window = np.linspace(flagged_day, flagged_day_2, num = steps)
                phase_obs = np.append(phase_obs, window)
                #flagging phases and adding them to an array 
                flagged_phase = np.append(flagged_phase, self.phase[j])
                flagged_phase = np.append(flagged_phase, self.phase[j+1])
                print(flagged_day)
            #want to iterate through both of the arrays
            j += 1
            i += 1
            if i >= len(distance): 
                break
        #checking to see if there are doubles in the phase and removing them
        for i in range(len(phase_obs)): 
            if i == len(phase_obs) - 1: 
                break
            if phase_obs[i] == phase_obs[i-1]: 
                phase_obs = np.delete(phase_obs, i)
        #calculating the days from conjunction the flagged phases are at since the empheremis is 
        #based on the first planet, we are using the first planet's period and Tp
        self.period_days = period_days
        self.phase_obs = phase_obs
        self.flagged_phase = flagged_phase
        return len(self.flagged_phase), self.flagged_phase, self.phase_obs
    
    #coverts the semester into JD
    def JD_convert(self, semester):
        #ex of semester would be 2023A or 2023B
        self.semester = semester 
        if semester[4] == 'A': 
            #beginning of A semester is 2/1, end is 7/31
            semester_year = int(semester.split('A')[0])
            year_beg = int(semester_year) 
            year_end = int(semester_year)
            date_beg = f"{year_beg}-02-01"
            date_end = f"{year_end}-07-31"
        elif semester[4] == 'B': 
            #beginning of B semester is 8/1, end is 1/31/year+1
            semester_year = semester.split('B')[0]
            year_end = int(semester_year+1)
            year_beg = int(semester_year) 
            date_beg = f"{year_beg}-08-01"
            date_end = f"{year_end}-01-31"
        else: 
            print("Please enter a valid semester.")
            
        
        #getting all of the possible JD dates for observation during this semester by using astropy.Time
        date_beg = Time(date_beg)
        JD_beg = date_beg.jd
        self.JD_beg = JD_beg
        
        date_end = Time(date_end)
        JD_end = date_end.jd
        self.JD_end = JD_end
        
        #creating array of every single day in semester which varies based on semester
        if year_end != year_beg: 
            possible_days = np.linspace(JD_beg, JD_end, num = 183)
        else: 
            possible_days = np.linspace(JD_beg, JD_end, num = 180)
        self.possible_days = possible_days
        
        return self.JD_beg, self.JD_end
    
    #the next function will be able to tell you which cadence is the best to observe in 
    def cadence_optimize(self): 
        #this will give the times within the semester where the time of conjunction will occur 
        future_tp = self.tp
        print(self.tp, future_tp, self.JD_beg, self.JD_end, self.period)
        list_future_tp = []
        while future_tp < self.JD_end:
            future_tp += self.period
            #creates list of times where future conjunction will occur within semester 
            if future_tp < self.JD_end and future_tp >= self.JD_beg: 
                list_future_tp = np.append(list_future_tp, future_tp)
            elif future_tp >= self.JD_end:
                break
            else: 
                continue
        #adding the days where you should observe from days of conjunction to the future days of conjunction 
        observe_times = []
        for i in range(len(list_future_tp)): 
            #making sure that the observed days are within the semester user input
            if list_future_tp[i] + self.phase_obs.any() > self.JD_end: 
                break
            else: 
                obs_date = list_future_tp[i] + self.phase_obs
                observe_times = np.append(observe_times, obs_date)
                
        #days where user should observe
        return list_future_tp, observe_times
