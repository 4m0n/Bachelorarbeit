import pandas as pd
from os import listdir
import os
from os.path import isfile, join
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import numpy as np
import math
import time
from numba import jit
from tqdm import tqdm
from scipy import optimize
import yaml
import scipy.signal as signal
from astropy.timeseries import LombScargle
import numpy as np
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq


from rich import print
from rich.traceback import install
from rich.console import Console
install()
console = Console()

def fit_func_sin(x, a, b, c, d):
    return a * np.sin(b * x + c) + d 

def fit(x, y):
    amplitude_guess = np.std(y) * 2**0.5
    frequency_guess = 2 * np.pi / (x[-1] - x[0])
    phase_guess = 0
    offset_guess = np.mean(y)
    p0 = [amplitude_guess, frequency_guess, phase_guess, offset_guess]
    
    params, params_covariance = optimize.curve_fit(fit_func_sin, x, y, p0=p0, maxfev=100000)
    return params



def maxima(frequency, power):
    peaks, properties = find_peaks(power, height=0.01)
    console.print(f"PAEAKS: {peaks} Properties: {properties}")  
    console.print(f"PEAKS: {frequency[peaks]*(2*np.pi)},")
        
    


rand = np.random.default_rng(42)
timerange = 10 * 31_536_000
t = timerange * rand.random(10000)
indices_to_remove = []
t = np.sort(t)
for i in range(0, len(t), 1500):
    indices_to_remove.extend(range(i, min(i + 500, len(t))))
t = np.delete(t, indices_to_remove)

#t = np.concatenate((t[0:1000],t[1500:-1]))
print(len(t))
t2 = np.linspace(0, timerange, 10000)
if 1: 
    y = 0.8*np.sin(10/timerange * t) + 1* rand.standard_normal(len(t)) - 0.5
    console.print(f"Schwingung1: {10/timerange}")
if 0:
    y = 0.1*np.sin(10/timerange * t) +0.1*np.sin(20/timerange * t) + 0.1*np.cos(800/timerange*t) + 1* rand.standard_normal(len(t)) - 0.5
    console.print(f"Schwingung1: {10/timerange} Schwingung2: {20/timerange} Schwingung3: {800/timerange}, : {1/(800/timerange)/(60*60*24)}")
if 0:
    y = 1 * np.exp(-(((t - timerange/2))/(2 * timerange/1000))**2) #+ 0.2*rand.standard_normal(10000)
if 0:
    y = 1* rand.standard_normal(len(t))


params = fit(t,y)
console.print(f"Periode: {params[1]}")

fig, (ax_t, ax_f) = plt.subplots(2, 1, constrained_layout=True)


y_fit = fit_func_sin(t, *params)
ax_t.plot(t, y_fit, color='green', label='Fitted Curve', zorder = 10)
ax_t.plot(t, y, 'b+')

#ax_t.plot(t2, np.sin(10 / timerange * t2), color='red')
ax_t.plot(t2, np.sin(20 / timerange * t2), color='red')
#ax_t.plot(t2, np.sin(800 / timerange * t2), color='red')
ax_t.grid(True, which="both", linestyle="--", linewidth=0.5)


# Plot für Lomb-Scargle
frequency, power = LombScargle(t, y).autopower(maximum_frequency = 1e-6, samples_per_peak = 20)
print(f"max Power: {max(power)}")
if max(power) >= 0.003:
    #power  = power / max(power)
    pass
maxima(frequency, power)
ax_f.plot(frequency, power)
ax_f.set_xlabel('Frequency [Hz]] Astropy')
# ax_f.vlines(10/timerange/(2*np.pi),min(power),max(power) ,color='red')
# ax_f.vlines(20/timerange/(2*np.pi), min(power),max(power), color='green')
ax_f.set_ylabel('Normalized amplitude')
ax_f.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.show()
