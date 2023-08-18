import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
import lightkurve
from matplotlib.pyplot import cm
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as plticker

hiresdata = pd.read_csv('T001246_4pl_data.csv')
harps = pd.read_csv('HARPSN_2/TOI-1246-tng_harpn-0028-srv-rvs.dat',delimiter='     ',header=None)
harps = harps[[0,1,2]]
harps.columns = ['time', 'mnvel', 'errvel']

data = pd.concat([hiresdata,harps])
t = np.array(data.time)
vel = np.array(data.mnvel)
errvel = np.array(data.errvel)
periods = np.array([4.307,5.904,18.66,37.918, 93])

old_data = pd.read_pickle("15092022rdvel+harps_4pl_no18d_efree/rdvel+harps_post_obj.pkl")
new_data = pd.read_pickle("rdvel+harps_finalfinal/rdvel+harps_post_obj.pkl")

#read in model from radvel 
old_data = pd.read_pickle("31072021rdvel+harps/rdvel+harps_post_obj.pkl")
new_data = pd.read_pickle("Model Comparison Radvel/06102021_4pl/rdvel+harps_post_obj.pkl")
data93 = pd.read_pickle("17092021rdvel+harps_5thplanet_93d/rdvel+harps_post_obj.pkl")
modelt = np.linspace(min(t), max(t),num=1000)

fig, (ax,ax2) = plt.subplots(2,1,gridspec_kw={'height_ratios': [8,3]}, sharex=True)
ax.plot(modelt, data.model(modelt), lw=1, c='k')
ax.errorbar(t, vel, yerr=errvel,ms=5, fmt='o')
ax.hlines(0,min(t), max(t), linestyles='dashed')
ax.set_xlim(min(t), max(t))
ax.set_ylabel('RV')
ax2.errorbar(t, vel-data.model(t), ms=5, fmt='o')
ax2.hlines(0,min(t), max(t), linestyles='dashed')
ax2.set_xlim(min(t), max(t))
ax2.set_xlabel('Time')

#actual data - data points from the model
oldres = vel-old_data.model(t)
newres = vel-new_data.model(t)
res93 = vel-data93.model(t)
res_err = errvel
f = np.linspace(0.002,2, 10000)
oldp = LombScargle(t,oldres, res_err).power(f)
newp = LombScargle(t,newres, res_err).power(f)
p93 = LombScargle(t,res93, res_err).power(f)

fig, ax = plt.subplots(1,1,figsize=(5,2))
ax.plot(1/f,oldp,c='k',linewidth=0.7, label='3pl_no18d_efree')
ax.set_ylim(0,0.3)
ax.set_xscale('log')
ax.xaxis.set_major_formatter(plticker.ScalarFormatter())
ax.set_xticks([1, 3, 5, 10, 30, 50, 100, 300, 500, 1000])
ax.set_xlim(0.1,500)
colour = iter(cm.rainbow(np.linspace(0,1,len(periods))))
for i, per in enumerate(periods):
    c = next(colour)
    ax.vlines(x=per,ymin=0.1,ymax=0.35,colors=c,lw=1,linestyle='dashed')
plt.savefig('15092022rdvel+harps_4pl_no18d_efree_residuals_periodogram.png', dpi=200)


fig, (ax,ax2) = plt.subplots(2,1, sharex=True)
ax.plot(1/f,oldp,c='k',linewidth=0.7, label='+New')
ax.set_ylim(0,0.3)
ax.set_xscale('log')
ax.legend()
ax2.plot(1/f,newp,c='k',linewidth=0.7, label='Old')
ax2.set_xlabel('Period [d]')
ax2.set_ylim(0,0.3)
ax2.set_xscale('log')
ax2.legend()
ax2.xaxis.set_major_formatter(plticker.ScalarFormatter())
ax2.set_xticks([1, 3, 5, 10, 30, 50, 100, 300, 500, 1000])
ax.set_xlim(1,500)
ax2.set_xlim(1,500)
colour = iter(cm.coolwarm(np.linspace(0,1,len(periods))))
for i, per in enumerate(periods):
    c = next(colour)
    for ax1 in [ax, ax2]:
        ax1.vlines(x=per,ymin=0.2,ymax=0.35,colors=c,lw=2,linestyle='dashed')
        if ax1 == ax:
            ax1.text(x=per-[1.2,1.5,4,10,10,1,1][i], y=0.35, s='{}'.format(['.02','.03','.01','.04','',''][i]), fontsize=16, c=c)
plt.savefig('05092022_residuals_periodogram.png', dpi=200)

best_f = f[np.argmax(p)]
t_fit = np.linspace(0,1)
ls = LombScargle(t,res,res_err)
y_fit = ls.model(t_fit, best_f)

lc = lightkurve.LightCurve(time=t, flux=res, flux_err=res_err)
lc.fold(1/best_f).scatter()()
