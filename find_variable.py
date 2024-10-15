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

# ========== Interesting Galaxies ==========

gal1 = ["NGC 1566", "NGC 4593", "NGC 4131"]
gal2 = ["ESO253-G003"]
gal3 = ["NGC 4826", "NGC 2258", "ESO 506-G05","NGC 3783","NGC 5899", "NGC 4748", "NGC 3147","PKS 0131-522","NGC 5078","NGC 5291","NGC 3227","NGC 428","NGC 404", "NGC 3962", "ESO 512-G20","UGC 6728","NGC 1068", "MARK 391","NGC 2617","NGC 676",
        "IC 4296","ESO 362-G18","NGC 4105","ESO 420-G13","NGC 1241", "NGC 5033", "NGC 7552","NGC 4700","NGC 7590", "UGC 3478", "NGC 2639","IC 1459","NGC 3281", "M 31", "NGC 2217","NGC 5005","NGC 7410","ESO 373-G29","NGC 4051","NGC 4736","NGC 4278",
        "NGC 3516","NGC 3660","NGC 4725","ESO 377-G24","NGC 7213","MARK    6", "Mark 766","NGC 5377","NGC 1566","CTS J13.12","NGC 4594","NGC 4151","NGC 5273","NGC 2681","NGC 6814","NGC 4411B","NGC 4395","IGR J16024-6107","NGC 4593","NGC 4579","NGC 3941",
        "NGC 1365","NGC 7714","NGC 4235","POX 52", "NGC 1204"]
gal4 = ["NGC 3783","NGC 3227","ESO 512-G20","NGC 1068","NGC 2617","NGC 676","ESO 362-G18", "UGC 3478","NGC 4736","NGC 3516","MARK    6","Mark 766","NGC 1566","CTS J13.12","NGC 4594","NGC 4151","NGC 5273","IGR J16024-6107","NGC 4593","NGC 1365"]
# ========== Tresholds ==========

PlotAktiv = True
treshold_F_var = 0.0025
treshold_R = 1.2

# ========== Variables ==========

show_galaxies = gal4 # sonst == []

# ========== Paths ==========

value = "Flux"
path = "activity_curves/"
load_path = "final_light_curves/"
activity_path = "light_curves/active_galaxies.csv"

class Plots:
    def show_plots():
        galaxy_active = pd.read_csv(activity_path)
        files = [f for f in listdir(load_path) if isfile(join(load_path, f))]
        show_galaxies_lower = [item.replace(" ","").lower()for item in show_galaxies]
        for file in tqdm(files):
            name = file[:-4].replace(" ","")
            if PlotAktiv:
                if len(show_galaxies) > 0:
                    if name.lower() not in show_galaxies_lower:
                        continue
                if file[:-4] in galaxy_active["name"].values:
                    if galaxy_active.loc[galaxy_active["name"] == file[:-4], "acivity"].values[0] <= treshold_F_var or galaxy_active.loc[galaxy_active["name"] == file[:-4], "R"].values[0] <= treshold_R:
                        continue

                Plots.plot_curves(file[:-4])

        
        
        
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
            if y[i] > R_treshold and F_treshold < x[i]:
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
    def plot_curves(name):

        file = FileManager.load_data(name)
        x, y, cam = file.index.copy(), file[value].copy(), file["Camera"].copy()
        file2 = BasicCalcs.rolling_mid(file,"30D")
        x1, y1 = file2.index.copy(), file2[value].copy()
        # Kamera farben
        
        farben = [
        "blue", "black", "cyan", "magenta", "yellow", "black", "white", "orange", 
        "purple", "brown", "pink", "gray", "olive", "darkblue", "lime", "indigo", 
        "gold", "darkgreen", "teal", "black", "cyan", "magenta", "yellow", "black", "white", "orange", 
        "purple", "brown", "pink", "gray", "olive", "darkblue", "lime", "indigo", 
        "gold", "darkgreen", "teal"
        ]
        cameras = cam.unique()  
        c2 = []
        for i in cam:
            c2.append(farben[np.where(cameras == i)[0][0]])
            
        c3 = []
        c4 = []
        for index, i in enumerate(cam):
            if file["Filter"].iloc[index] == "V":
                c3.append(farben[np.where(cameras == i)[0][0]])
            else:
                c4.append(farben[np.where(cameras == i)[0][0]])
                
        x_1 = file.loc[file["Filter"] == "V", :].index.copy()
        x_2 = file.loc[file["Filter"] == "g", :].index.copy()
        y_1 = file.loc[file["Filter"] == "V", value].values.copy()
        y_2 = file.loc[file["Filter"] == "g", value].values.copy()
        # === Plot
        galaxy_active = pd.read_csv(activity_path)
        try:
            if name in galaxy_active["name"].values:
                if galaxy_active.loc[galaxy_active["name"] == name, "acivity"].values[0] <= treshold_F_var and galaxy_active.loc[galaxy_active["name"] == name, "R"].values[0] <= treshold_R:
                    plt.title(f"NICHT VARIABEL Galaxy: {name}, cuts: {galaxy_active.loc[galaxy_active['name'] == name, 'cuts'].values[0]} \nTH F: {treshold_F_var} Activity F: {round(galaxy_active.loc[galaxy_active['name'] == name, 'acivity'].values[0]*10000)/10000}\nTR R: {treshold_R} Activity R: {round(galaxy_active.loc[galaxy_active['name'] == name, 'R'].values[0]*100)/100}")
                else:
                    if galaxy_active.loc[galaxy_active['name'] == name, 'R'].values[0] == np.inf or galaxy_active.loc[galaxy_active['name'] == name, 'acivity'].values[0] == np.inf:
                        try:
                            plt.title(f"VARIABEL Galaxy: {name}, cuts: {galaxy_active.loc[galaxy_active['name'] == name, 'cuts'].values[0]} \nTH F: {treshold_F_var} Activity F: {round(galaxy_active.loc[galaxy_active['name'] == name, 'acivity'].values[0]*10000)/10000}\nTR R: {treshold_R} Activity R: inf")
                        except:
                            plt.title(f"VARIABEL Galaxy: {name}, cuts: {galaxy_active.loc[galaxy_active['name'] == name, 'cuts'].values[0]} \nTH F: {treshold_F_var} Activity F: Inf\nTR R: {treshold_R} Activity R: inf")

                    else:
                        plt.title(f"VARIABEL \nGalaxy: {name}, cuts: {galaxy_active.loc[galaxy_active['name'] == name, 'cuts'].values[0]} \nTH F: {treshold_F_var} Activity F: {round(galaxy_active.loc[galaxy_active['name'] == name, 'acivity'].values[0]*10000)/10000}\nTR R: {treshold_R} Activity R: {round(galaxy_active.loc[galaxy_active['name'] == name, 'R'].values[0]*100)/100}")
            else:
                plt.title(f"Galaxy: {name} - nicht gefunden")
        except:
            plt.title(f"Galaxy: {name} - nicht gefunden + Error")
            
        
        plt.plot(x1,y1,zorder=10, label="30 Tage", color = "red")
        plt.scatter(x_1,y_1,c = c3, alpha=0.2, zorder=5, marker = "x") # plot verschobene orginalpunkte
        plt.scatter(x_2,y_2,c = c4, alpha=0.2, zorder=5, marker = "o") # plot verschobene orginalpunkte
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()


class FileManager:
    def load_data(file):
        data = pd.read_csv(load_path + file + ".csv", index_col = 0)
        #print(f"Test:\nIndex:\n{data.index}\nFlux:\n{data["Flux"]}\nFlux Error:\n{data["Flux Error"]}\nCamera\n{data["Camera"]}")
        data.index = pd.to_datetime(data.index)
        return data    
    
    def load_cuts(name):
        activity = pd.read_csv(activity_path)
        if name in activity["name"].values:
            cuts = activity.loc[activity['name'] == name, 'cuts'].values[0]
        else: cuts = -1
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
        try:
            sum = np.sqrt(sum/length)
        except:
            sum = 0
        return sum # nicht quadriert
    def rolling_mid(file, rolling_time ="28D"):
        file2 = file.copy() 
        file2.sort_index(inplace=True)
        file2[value] = file2[value].rolling(window=rolling_time, center = False, min_periods = 4).mean()
        return file2

class FindActive:
    def peak_to_peak_amplitudes(name): # returns R
        curve = FileManager.load_data(name)
        file2 = BasicCalcs.normalize(curve)
        #print(f"Max: {file2[value].max()} Min: {file2[value].min()}, Zsm: {file2[value].max() / file2[value].min()}")
        if file2[value].max() / file2[value].min() <= 0:
            print(f"\nALARM {file2[value].max() / file2[value].min()}\n")
        return file2[value].max() / file2[value].min()
    
    def fractional_variation(name):
        curve = FileManager.load_data(name)
        file2 = curve.copy()
        file2 = BasicCalcs.normalize(file2)
        activity = (file2[value].std()**2-BasicCalcs.delta(file2)**2) / file2[value].mean()
        return activity
    
    def jumps(name):
        curve = FileManager.load_data(name)
        file2 = BasicCalcs.normalize(curve)
               
        
        
    def stetig(name):
        curve = FileManager.load_data(name)
        file2 = BasicCalcs.normalize(curve)
        

    
def start():    
    with open(path+"new_active_galaxies.csv", 'w') as datei:
        datei.write("name,activity,R,activity*R,cuts\n")
        
    files = [f for f in listdir(load_path) if isfile(join(load_path, f))]
    for file in tqdm(files):
        FileManager.group_galaxies(file[:-4])
Plots.show_plots()

#Plots.standart1()   
#start()