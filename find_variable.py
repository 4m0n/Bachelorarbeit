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

# ========== Interesting Galaxies ==========

gal1 = ["NGC 1566", "NGC 4593", "NGC 4131"]
gal2 = ["ESO253-G003"]
gal3 = ["NGC 4826", "NGC 2258", "ESO 506-G05","NGC 3783","NGC 5899", "NGC 4748", "NGC 3147","PKS 0131-522","NGC 5078","NGC 5291","NGC 3227","NGC 428","NGC 404", "NGC 3962", "ESO 512-G20","UGC 6728","NGC 1068", "MARK 391","NGC 2617","NGC 676",
        "IC 4296","ESO 362-G18","NGC 4105","ESO 420-G13","NGC 1241", "NGC 5033", "NGC 7552","NGC 4700","NGC 7590", "UGC 3478", "NGC 2639","IC 1459","NGC 3281", "M 31", "NGC 2217","NGC 5005","NGC 7410","ESO 373-G29","NGC 4051","NGC 4736","NGC 4278",
        "NGC 3516","NGC 3660","NGC 4725","ESO 377-G24","NGC 7213","MARK    6", "Mark 766","NGC 5377","NGC 1566","CTS J13.12","NGC 4594","NGC 4151","NGC 5273","NGC 2681","NGC 6814","NGC 4411B","NGC 4395","IGR J16024-6107","NGC 4593","NGC 4579","NGC 3941",
        "NGC 1365","NGC 7714","NGC 4235","POX 52", "NGC 1204"]
gal4 = ["NGC 3783","NGC 3227","ESO 512-G20","NGC 1068","NGC 2617","NGC 676","ESO 362-G18", "UGC 3478","NGC 4736","NGC 3516","MARK    6","Mark 766","NGC 1566","CTS J13.12","NGC 4594","NGC 4151","NGC 5273","IGR J16024-6107","NGC 4593","NGC 1365"]
periodic = ["NGC 4826", "NGC 5899", "NGC 4748","NGC 3147","NGC5291","NGC 3227","NGC 404","NGC 3962","ESO 512-G20","UGC 6728", "NGC 676","IC 4296","NGC 5033","NGC 7590","UGC 3478","NGC2639","IC 1459","NGC 3281","M 31","NGC 2217","NGC 5005","ESO 373-G29",
            "NGC 4736","NGC 3516","NGC 3660","NGC 4725","ESO 377-G24","MARK 6", "MARK 766","CTS J13.12","NGC 4594","NGC 4151","NGC 5273","NGC 2681","NGC 6814","IGR J16024-6107",
            "NGC 4579","NGC 1365","NGC 7714","POX 52"]
peak = ["PKS 0131-522","NGC 1068","NGC 2617","NGC 1566","NGC 4411B","NGC 3941"]
others = ["NGC 3783","NGC 428","ESO 362-G18","NGC 4736","NGC 3516","MARK 6", "NGC 4151","IGR J16024-6107","NGC 4593","NGC 1204"] 

# ========== thresholds ==========

PlotAktiv = True
F_threshold = 0.0025
R_threshold = 1.2
amplitude_threshold = 0.1
amp_diff_threshold = 0.2
T_threshold = [580_000, 5e8]
F_and_R_threshold = 1

# ========== Variables ==========

show_galaxies = gal3 # sonst == []

# ========== Paths ==========

value = "Flux"
path = "activity_curves/"
load_path = "final_light_curves/"
activity_path = "light_curves/active_galaxies.csv"


class Conditions:
    def main(*,R=0,F=0,amp_diff=0,T=0,Dt = 5e8):
        return Conditions.periodic(amp_diff,T,Dt)
    def F_var_R(R,F):
        return R > R_threshold and F_threshold < F
    def periodic(amp_diff,T,Dt):
        return amp_diff > amp_diff_threshold and T > T_threshold[0] and T/(60*60*24)*365 < Dt*2
 

class Plots:
    def show_plots():
        galaxy_active = pd.read_csv(path+"new_active_galaxies.csv")
        files = [f for f in listdir(load_path) if isfile(join(load_path, f))]
        show_galaxies_lower = [item.replace(" ","").lower()for item in show_galaxies]
        for file in tqdm(files):
            name = file[:-4].replace(" ","")
            if PlotAktiv:
                if len(show_galaxies) > 0 and False:
                    if name.lower() not in show_galaxies_lower:
                        continue
                elif file[:-4] in galaxy_active["name"].values:
                    amp_diff = galaxy_active.loc[galaxy_active["name"] == file[:-4], "amp_diff"].values[0]
                    T = galaxy_active.loc[galaxy_active["name"] == file[:-4], "period"].values[0]
                    F = galaxy_active.loc[galaxy_active["name"] == file[:-4], "activity"].values[0]
                    R = galaxy_active.loc[galaxy_active["name"] == file[:-4], "R"].values[0]
                    Dt = galaxy_active.loc[galaxy_active["name"] == file[:-4], "Dt"].values[0]
                    if not Conditions.main(R = R,F = F,amp_diff = amp_diff,T = T, Dt = Dt):
                        continue
                Plots.plot_curves(file[:-4])

        
        
        
    def standart1():
        path1 = path+"new_active_galaxies.csv"
        name = np.loadtxt(path1, delimiter=',', skiprows=1,usecols=0,dtype=str)
        F_var = np.loadtxt(path1, delimiter=',', skiprows=1,usecols=1) #F_var
        R = np.loadtxt(path1, delimiter=',', skiprows=1,usecols=2) #R
        F_and_R = np.loadtxt(path1, delimiter=',', skiprows=1,usecols=3)
        cuts = np.loadtxt(path1, delimiter=',', skiprows=1,usecols=4) #cuts
        amplitude = np.loadtxt(path1, delimiter=',', skiprows=1,usecols=5) #amplitude
        amp_diff = np.loadtxt(path1, delimiter=',', skiprows=1,usecols=6) #amplitude_diff
        T = abs(np.loadtxt(path1, delimiter=',', skiprows=1,usecols=7))
        Dt = np.loadtxt(path1, delimiter=',', skiprows=1,usecols=8) # delta T
        
        x = "amp_diff"
        y = "T"
        z, z_title = cuts, "cuts"
        
        def setup(x):
            match x:
                case "F_var":
                    x = F_var
                    x_title = "F_var"
                    threshold_x = F_threshold
                case "R":
                    x = R
                    x_title = "R"
                    threshold_x = R_threshold
                case "amplitude":
                    x = amplitude
                    x_title = "amplitude"
                    threshold_x = amplitude_threshold
                case "amp_diff":
                    x = amp_diff
                    x_title = "amp_diff"
                    threshold_x = amp_diff_threshold
                case "T":
                    x = T
                    x_title = "T"
                    threshold_x = T_threshold
                case "F_and_R":
                    x = R-(F_var*1000)
                    x_title = "F_and_R"
                    threshold_x = F_and_R_threshold
            return x, x_title, threshold_x
        
        x, x_title, threshold_x = setup(x)
        y, y_title, threshold_y = setup(y)    
        x_target = []
        y_target = []
        
        x_thresh = []
        y_thresh = []
        
        
        if len(show_galaxies) > 0:
            for i in reversed(range(len(name))):
                if name[i] in  show_galaxies:
                    x_target.append(x[i])
                    y_target.append(y[i])                    
                    
        aktive_prozent = 0
        variabel = pd.DataFrame(columns=["name","F","R"])
        for i in range(len(y)):
            if Conditions.main(R = R[i],F = F_var[i],amp_diff = amp_diff[i],T = T[i],Dt = Dt[i]):
                aktive_prozent += 1
                x_thresh.append(x[i])
                y_thresh.append(y[i])

        aktive_prozent = round(aktive_prozent / len(y) * 10000)/100
        print(f"Variable Galaxien nach aktuellem Filter:\n{variabel.to_string()}")
        if type(threshold_x) == type(4.20):
            threshold_x = [threshold_x]
        if type(threshold_y) == type(4.20):
            threshold_y = [threshold_y]
        for i in threshold_y:
            plt.hlines(i,min(x),max(x)) # R
        for i in threshold_x:    
            plt.vlines(i,min(y),max(y)) # F
        
        plt.scatter(x,y,c = z,s = 50, edgecolor = "k", alpha = 0.5,zorder = 1)
        plt.scatter(x_target,y_target,c = "red",s = 50, edgecolor = "k", alpha = 0.8,marker = 5,zorder = 2)
        plt.scatter(x_thresh,y_thresh,c = "blue",s = 50, edgecolor = "k", alpha = 0.5,marker = 4,zorder = 1)
        
        plt.colorbar(label=z_title)
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        if len(y_target)<1:
            prozent_korrekte_vorhersagen = -1
        else:
            prozent_korrekte_vorhersagen = round(len(np.intersect1d(y_target, y_thresh))/len(y_target)*10000)/100
        if len(y_thresh)<1:
            prozent_treffer_in_filter = -1
        else:
            prozent_treffer_in_filter = round(len(np.intersect1d(y_thresh, y_target))/len(y_thresh)*10000)/100
        plt.title(f"Anzahl: {len(x)}, Variabel: {aktive_prozent}%\nFilter trifft {prozent_korrekte_vorhersagen}% der Galaxien\nKorrekt in Filter: {prozent_treffer_in_filter}%")
        plt.grid()
        plt.show()
        plt.close()
        
    def plot_curves(name):

        file2 = FileManager.load_data(name)
        file = BasicCalcs.normalize_null(file2)
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
        galaxy_active = pd.read_csv(path + "new_active_galaxies.csv")
        
        cuts = galaxy_active.loc[galaxy_active['name'] == name, 'cuts'].values[0]
        
        F = galaxy_active.loc[galaxy_active["name"] == name, "activity"].values[0]
        R = galaxy_active.loc[galaxy_active["name"] == name, "R"].values[0]
        amp_diff = galaxy_active.loc[galaxy_active["name"] == name, "amp_diff"].values[0]
        T = galaxy_active.loc[galaxy_active["name"] == name, "period"].values[0]
        Dt = galaxy_active.loc[galaxy_active["name"] == name, "Dt"].values[0]
        try:tr_F_var = round(F*10000)/10000
        except:tr_F_var = np.inf
        try: tr_R = round(R*100)/100
        except: tr_R = np.inf
        try: tr_amplitude = round(amp_diff*100)/100
        except: tr_amplitude = np.inf
        if name in galaxy_active["name"].values:
            if not Conditions.main(F=F,R=R,amp_diff=amp_diff,T=T,Dt=Dt):
                plt.title(f"NICHT VARIABEL Galaxy: {name}, cuts: {cuts} \nTH F: {F_threshold} Activity F: {tr_F_var}\nTR R: {R_threshold} Activity R: {tr_R}\nTR Amp: {amp_diff_threshold} Amp: {tr_amplitude}, T = {round(T/86400)} Jahre")
            else:
                plt.title(f"VARIABEL \nGalaxy: {name}, cuts: {cuts} \nTH F: {F_threshold} Activity F: {tr_F_var}\nTR R: {R_threshold} Activity R: {tr_R}\n TR Amp: {amp_diff_threshold} Amp: {tr_amplitude}, T = {round(T/86400)} Jahre")
        else:
            plt.title(f"Galaxy: {name} - nicht gefunden")

            
        
        plt.plot(x1,y1,zorder=10, label="30 Tage", color = "red")
        plt.scatter(x_1,y_1,c = c3, alpha=0.2, zorder=5, marker = "x") # plot verschobene orginalpunkte
        plt.scatter(x_2,y_2,c = c4, alpha=0.2, zorder=5, marker = "o") # plot verschobene orginalpunkte
        x_temp,y_temp,_,_,_ = FindActive.periodic(name)
        plt.plot(x_temp,y_temp, label = "Sinus Fit", color = "green")
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
        _,_,amplitude,amp_diff,period = FindActive.periodic(name)
        deltaT = BasicCalcs.TimeDifference(name)
        
        if os.path.isfile(file_path) == False:
            with open(file_path, 'w') as datei:
                datei.write("name,activity,R,activity*R,cuts,amplitude,amp_diff,period,Dt\n")
        with open(file_path, 'a') as datei:
            datei.write(f"{name},{Fvar},{R},{Fvar*R},{cuts},{amplitude},{amp_diff},{period},{deltaT.days}\n")
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
    
    def normalize2(file):
        curve = file.copy()
        shift = curve[value].median() 
        curve[value] = curve[value] - shift
        curve[f"{value} Error"] = curve[f"{value} Error"]/curve[value].max()
        curve[value] = curve[value]/curve[value].max()
        curve[value] = curve[value] - curve[value].min() + 1
        
        return curve
        
    def normalize_sin(file):
        curve = file.copy()
        shift = curve[value].mean() 
        
        curve[value] = curve[value] - shift
        curve[value] = curve[value]/file[value].max()
        return curve
    
    def normalize_null(file):
        curve = file.copy()
        curve[value] = curve[value] - curve[value].min() 
        curve[value] = curve[value]/curve[value].max()
        return curve
    def TimeDifference(name):
        curve = FileManager.load_data(name)
        return curve.index[-1] - curve.index[0]
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
    
    def fit_func_sin(x, a, b, c, d):
        return a * np.sin(b * x + c) + d 
    
    def Datetime_in_Unix(date):
        unix = []
        for i in date:
            unix.append(i.timestamp())
        return unix
    
    def Unix_in_Datetime(date):
        datetime = []
        for i in date:
            datetime.append(pd.to_datetime(i, unit='s'))
        return datetime

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
    
    def periodic(name):
        curve = FileManager.load_data(name)
        file2 = BasicCalcs.rolling_mid(curve)
        file2 = BasicCalcs.normalize_null(curve)
        #file2 = BasicCalcs.rolling_mid(file2)
        file2.dropna(inplace=True)
        maximum = curve[value].max()
        numeric_index = BasicCalcs.Datetime_in_Unix(file2.index)
        
        # ===== FIT =====
        if len(numeric_index) < 2:
            return [],[]
        time_diff = (numeric_index[-1] - numeric_index[0]) 
        
        amp = curve[value].max() - curve[value].min()
        params, params_covariance = optimize.curve_fit(BasicCalcs.fit_func_sin, numeric_index, file2[value].values, p0=[amp, 1/time_diff, 0, file2[value].mean()],maxfev=100000) # a * np.sin(b * x + c) + d 

        x = np.linspace(min(numeric_index), max(numeric_index), 10000)
        y = BasicCalcs.fit_func_sin(x, *params)
        x = BasicCalcs.Unix_in_Datetime(x)
        
        return x,y, abs(params[0]), (y.max() - y.min()),(1/abs(params[1]*60)) # T in sekunden       
    
    def jumps(name):
        curve = FileManager.load_data(name)
        file2 = BasicCalcs.normalize(curve)
               
        
        
    def stetig(name):
        curve = FileManager.load_data(name)
        file2 = BasicCalcs.normalize(curve)
        

    
def start():    
    
    galaxy_active = pd.read_csv(path+"new_active_galaxies.csv")
    files = [f for f in listdir(load_path) if isfile(join(load_path, f))]
    show_galaxies_lower = [item.replace(" ","").lower()for item in show_galaxies]

    
    with open(path+"new_active_galaxies.csv", 'w') as datei:
        datei.write("name,activity,R,activity*R,cuts,amplitude,amp_diff,period,Dt\n")
        
    files = [f for f in listdir(load_path) if isfile(join(load_path, f))]
    for file in tqdm(files):
        FileManager.group_galaxies(file[:-4])
#start()
#Plots.show_plots()

Plots.standart1()   