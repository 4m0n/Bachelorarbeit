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


value = "Flux"
path = "activity_curves/"
load_path = "final_light_curves/"
activity_path = "light_curves/active_galaxies.csv"


class Plots:
    def standart1():
        F_treshold = 0.0025
        R_treshold = 1.2
        path = "activity_curves/new_active_galaxies.csv"
        x = np.loadtxt(path, delimiter=',', skiprows=1,usecols=1) #F_var
        y = np.loadtxt(path, delimiter=',', skiprows=1,usecols=2) #R
        z = np.loadtxt(path, delimiter=',', skiprows=1,usecols=4) #cuts
        times = np.loadtxt(path, delimiter=',', skiprows=1,usecols=3)
        name = np.loadtxt(path, delimiter=',', skiprows=1,usecols=0,dtype=str)
        aktive_prozent = 0
        variabel = pd.DataFrame(columns=["name","F","R"])
        for i in range(len(y)):
            if y[i] > R_treshold or F_treshold < x[i]:
                aktive_prozent += 1
                if variabel.empty:
                    variabel = pd.DataFrame({"name":[name[i]],"F":[x[i]],"R":[y[i]]})
                else:
                    variabel = pd.concat([variabel,pd.DataFrame({"name":[name[i]],"F":[x[i]],"R":[y[i]]})])
        aktive_prozent = round(aktive_prozent / len(y) * 10000)/100
        print(f"Variable Galaxien nach aktuellem Filter:\n{variabel.to_string()}")
        plt.hlines(R_treshold,min(x),max(x)) # R
        plt.vlines(F_treshold,min(y),max(y)) # F
        plt.scatter(x,y,c = z,s = 50, edgecolor = "k", alpha = 0.5)
        plt.colorbar(label='Cuts')
        plt.xlabel("F_var")
        plt.ylabel("R")
        plt.title(f"Anzahl: {len(x)}, Variabel: {aktive_prozent}%")
        plt.grid()
        plt.show()
        plt.close()


class FileManager:
    def load_data(file):
        data = pd.read_csv(load_path + file + ".csv", index_col = 0)
        #print(f"Test:\nIndex:\n{data.index}\nFlux:\n{data["Flux"]}\nFlux Error:\n{data["Flux Error"]}\nCamera\n{data["Camera"]}")
        return data    
    
    def load_cuts(name):
        activity = pd.read_csv(activity_path)
        cuts = activity.loc[activity['name'] == name, 'cuts'].values[0]
        return cuts

    def group_galaxies(name):
        file_path = path+"new_active_galaxies.csv"
        cuts = FileManager.load_cuts(name)
        Fvar = FindActive.fractional_variation(name)
        cuts = FileManager.load_cuts(name)
        R = FindActive.peak_to_peak_amplitudes(name)
        
        if os.path.isfile(file_path) == False:
            with open(file_path, 'w') as datei:
                datei.write("name,activity,R,activity*R,cuts\n")
        with open(file_path, 'a') as datei:
            datei.write(f"{name},{Fvar},{R},{Fvar*R},{cuts}\n")
        return
    
    
class BasicCalcs:
    def normalize(file):
        curve = file.copy()
        shift = 0
        if curve[value].min() <= 0:
            shift = curve[value].min() 
            curve[value] = curve[value] - shift + 1
            
        curve[f"{value} Error"] = curve[f"{value} Error"]/curve[value].max()
        curve[value] = curve[value]/curve[value].max()
        return curve
        
    def delta(file):
        sum = 0
        length = len(file[value])
        for i in range (length):
            sum += file[value + " Error"].iloc[i]**2
        sum = np.sqrt(sum/length)
        return sum # nicht quadriert


class FindActive:
    def peak_to_peak_amplitudes(name): # returns R
        curve = FileManager.load_data(name)
        file2 = BasicCalcs.normalize(curve)
        print(f"Max: {file2[value].max()} Min: {file2[value].min()}, Zsm: {file2[value].max() / file2[value].min()}")
        if file2[value].max() / file2[value].min() <= 0:
            print(f"\nALARM {file2[value].max() / file2[value].min()}\n")
        return file2[value].max() / file2[value].min()
    
    def fractional_variation(name):
        curve = FileManager.load_data(name)
        file2 = curve.copy()
        file2 = BasicCalcs.normalize(file2)
        activity = (file2[value].std()**2-BasicCalcs.delta(file2)**2) / file2[value].mean()
        return activity
    

    
def start():    
    with open(path+"new_active_galaxies.csv", 'w') as datei:
        datei.write("name,activity,R,activity*R,cuts\n")
        
    files = [f for f in listdir(load_path) if isfile(join(load_path, f))]
    for file in tqdm(files):
        FileManager.group_galaxies(file[:-4])
        
        
Plots.standart1()   
#start()