import pandas as pd
from os import listdir
import os
from os.path import isfile, join
import matplotlib.pyplot as plt
from astropy.units.quantity_helper.function_helpers import einsum
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
from astropy.time import Time
import csv

import numpy as np
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
from matplotlib.widgets import TextBox, Button,Slider
import ast


from rich import print
from rich.traceback import install
from rich.console import Console
install()
console = Console()


global Counter1
Counter1 = pd.DataFrame(columns=["Name","Value"])

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

def Datetime_in_Unix(date):
    unix = []
    for i in date:
        unix.append(i.timestamp())
    return unix

def normalize(file):
    value = "Flux"
    curve = file.copy()
    shift = 0
    if curve[value].min() <= 0:
        shift = curve[value].min()
        curve[value] = curve[value] - shift + 1

    curve[f"{value} Error"] = curve[f"{value} Error"] / curve[value].max()
    curve[value] = curve[value] / curve[value].max()
    return curve

def genereatePoints():
    rand = np.random.default_rng(42)
    timerange = 10 * 31_536_000
    t = timerange * rand.random(10000)
    indices_to_remove = []
    t = np.sort(t)
    for i in range(0, len(t), 1500):
        indices_to_remove.extend(range(i, min(i + 500, len(t))))
    t = np.delete(t, indices_to_remove)

    #t = np.concatenate((t[0:1000],t[1500:-1]))
    t2 = np.linspace(0, timerange, 10000)
    if 1:
        y = 0.8*np.sin(10/timerange * t) + 1* rand.standard_normal(len(t)) - 0.5
        #console.print(f"Schwingung1: {10/timerange}")
    if 0:
        y = 0.1*np.sin(10/timerange * t) +0.1*np.sin(20/timerange * t) + 0.1*np.cos(800/timerange*t) + 1* rand.standard_normal(len(t)) - 0.5
        console.print(f"Schwingung1: {10/timerange} Schwingung2: {20/timerange} Schwingung3: {800/timerange}, : {1/(800/timerange)/(60*60*24)}")
    if 0:
        y = 1 * np.exp(-(((t - timerange/2))/(2 * timerange/1000))**2) #+ 0.2*rand.standard_normal(10000)
    if 0:
        y = 1* rand.standard_normal(len(t))
    return t,t2,y,timerange


def rolling_mid(file, rolling_time="28D"):
    file.set_index("JD", inplace=True)
    file.index = pd.to_datetime(file.index)
    file2 = file.copy()
    file2.sort_index(inplace=True)
    file2["Flux"] = file2["Flux"].rolling(window=rolling_time, center=True, min_periods=4).mean()
    file2.reset_index(inplace=True)
    file2.dropna(subset=['Flux'],inplace = True)
    return file2

def loadCurve(val):
    file = pd.read_csv(f"final_light_curves/{val}.csv")
    file = normalize(file)
    #file = rolling_mid(file)
    t = []
    try:
        for val in file["JD"]:
            t.append(float(Time(val,format="iso",scale="utc").mjd))
    except:
        pass
    t = Datetime_in_Unix(pd.to_datetime(file["JD"]))
    mint = min(t)
    for i in range(len(t)):
        t[i] -= mint
    y = []
    for val in file["Flux"].values:
        y.append(val)

    return np.array(t),np.array(y)

def plot1(t,y, val = "None"):
    global timeSize
    global timePlot
    global vline
    global vline2
    global vline3
    global slider
    timePlot = 0.0
    vline = None
    vline2 = None
    vline3 = None
    slider = 0.0
    def submit(event):
        global vline
        global vline2
        global vline3
        global slider
        try:
            timePlot = float(timeSize.text)*1e8
            slider = float(SliderMid.val)
            if timePlot < 4:
                timePlot = 1.0/timePlot
        except:
            timePlot = 1
        # Remove the old vertical line if it exists
        if vline is not None:
            vline.remove()
            vline2.remove()
            vline3.remove()

        # Add the new vertical line and store its reference
        vline = ax_f.vlines(1/timePlot, min(power), max(power), color='red')
        vline2 = ax_t.vlines(0+slider, min(y), max(y), color='red')
        vline3 = ax_t.vlines(timePlot+slider,min(y), max(y), color='red')
        plt.draw()

    fig, (ax_t, ax_f) = plt.subplots(2, 1)
    plt.subplots_adjust(bottom=0.2)

    y_fit = fit_func_sin(t, *params)
    #ax_t.plot(t, y_fit, color='green', label='Fitted Curve', zorder = 10)
    ax_t.plot(t, y, 'b+')

    #ax_t.plot(t2, np.sin(10 / timerange * t2), color='red')
    try:
        pass
        #ax_t.plot(t2, np.sin(20 / timerange * t2), color='red')
    except:
        pass
    #ax_t.plot(t2, np.sin(800 / timerange * t2), color='red')
    ax_t.grid(True, which="both", linestyle="--", linewidth=0.5)


    # Plot für Lomb-Scargle
    frequency, power = LombScargle(t, y).autopower(maximum_frequency = 1e-7, samples_per_peak = 40)
    if max(power) >= 0.003:
        #power  = power / max(power)
        pass


    maxima(frequency, power)
    freq = pd.DataFrame({"Frequency": frequency, "Power": power})

    axbox1 = fig.add_axes([0.1, 0.01, 0.2, 0.05])
    timeSize = TextBox(axbox1, "Time:")
    axbox11 = fig.add_axes([0.4, 0.01, 0.2, 0.05])
    SliderMid = Slider(axbox11, "Shift:", valmin=0, valmax=max(t), valinit=0)
    axbox2 = fig.add_axes([0.71, 0.01, 0.1, 0.075])
    submit_button = Button(axbox2, "Submit")
    submit_button.on_clicked(submit)
    SliderMid.on_changed(submit)

    ax_f.plot(frequency, power)
    ax_f.set_xlabel('Frequency [Hz]] Astropy')
    ax_f.vlines(1e-8, min(power), max(power), color='black',alpha=0.5)
    #ax_f.vlines(2.9e-8, min(power), max(power), color='black', alpha=0.5)
    #ax_f.vlines(3.6e-8, min(power), max(power), color='black', alpha=0.5)
    ax_f.hlines(0.1, min(frequency), max(frequency), color='black',alpha=0.5)
    # ax_f.vlines(10/timerange/(2*np.pi),min(power),max(power) ,color='red')
    # ax_f.vlines(20/timerange/(2*np.pi), min(power),max(power), color='green')
    ax_f.set_ylabel('Normalized amplitude')
    ax_f.grid(True, which="both", linestyle="--", linewidth=0.5)
    global Counter1

    if len(Counter1) == 0:
        Counter1 = pd.DataFrame({"Name":[str(val)],"Value":[max(freq.loc[freq["Frequency"] >= 1e-8,"Power"])]})
    else:
        Counter1 = pd.concat([Counter1,pd.DataFrame({"Name":[val],"Value":[max(freq.loc[freq["Frequency"] >= 1e-8,"Power"])]})],ignore_index=True)
    #Counter1.append(max(freq.loc[freq["Frequency"] >= 1e-8,"Power"]))
    print(freq.loc[freq["Frequency"]>= 1e-8, "Power"].values)
    maxfreq = max(freq.loc[freq["Frequency"]>= 1e-8, "Power"].values)
    if ((freq["Frequency"] >= 1e-8) & (freq["Power"] >= 0.1)).any():
        ax_t.set_title(f"Yes {maxfreq}", color="green")
    else:
        ax_t.set_title(f"No {maxfreq}", color="red")
    plt.show()
    plt.close()

def histPlot(Counter1):
    plt.figure(figsize=(10, 6))
    plt.hist(Counter1["Value"].values, bins=500, color='skyblue', edgecolor='black')
    plt.vlines(0.1, ymin=0, ymax=20, color="red", linestyle='dashed', linewidth=1)

    plt.title('Verteilung: Fourier Amplituden Stärke', fontsize=16)
    plt.xlabel('Max Amplitude', fontsize=14)
    plt.ylabel('Anzahl', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.show()




# read from yml
with open("find_variable.yml") as f:
    config = yaml.safe_load(f)
# read all files
Galaxies = [f[:-4] for f in os.listdir("final_light_curves")]
Galaxies = ["PKS2158-380"]
Galaxies = config["ExampleGalaxies"]["periodic"]

if False:
    for val in tqdm(Galaxies):
        t,t2,y,timerange = genereatePoints()
        t,y = loadCurve(val)
        try:
            params = fit(t,y)
        except:
            pass
        plot1(t,y,val)

    #Counter1.reset_index(inplace=True)
    #Counter1.to_csv("Counter1.csv",index=False)

Counter1 = pd.read_csv("Counter1.csv")
Counter1 = Counter1[Counter1["Value"] < 1]
Counter1.sort_values(by="Value",ascending=False ,inplace=True)
print(Counter1)
print(f"Counter={Counter1.loc[Counter1["Value"] >= 0.3,'Name'].values.tolist()}")
print(f"Counter={len(Counter1.loc[Counter1["Value"] >= 0.3,'Name'].values)}")
plt.close()
histPlot(Counter1)