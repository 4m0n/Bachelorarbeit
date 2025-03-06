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
from scipy.interpolate import UnivariateSpline
import ast


from rich import print
from rich.traceback import install
from rich.console import Console
install()
console = Console()



with open("find_variable.yml") as f:
    config = yaml.safe_load(f)
# ========== thresholds ==========

F_threshold = config["Thresholds"]["F_threshold"]
R_threshold = config["Thresholds"]["R_threshold"]
amplitude_threshold = config["Thresholds"]["amplitude_threshold"]
amp_diff_threshold = config["Thresholds"]["amp_diff_threshold"] 
T_threshold = config["Thresholds"]["T_threshold"]
F_and_R_threshold = config["Thresholds"]["F_and_R_threshold"]

# ========== Variables ==========

show_galaxies = config["Plots"]["TargetGalaxies"] 

# ========== Paths ==========

value = config["Value"]
path = config["Paths"]["path"]
load_path = config["Paths"]["load_path"]
activity_path = config["Paths"]["activity_path"]

# ===========================

groups = pd.DataFrame({
    "function": ["periodic","linear","supernova","F_var_R","minPoints"],
    "count": [0,0,0,0,0]
})




class Conditions:
    def format_return_values(results):
        print(results)
        if type(results) == type(True) or len(results) > 2:
            return results
        formatted_results = []
        for boolean, value in results:
            if isinstance(value, float):  # Überprüfen, ob es eine Zahl ist
                # Formatieren auf 3 signifikante Stellen mit wissenschaftlicher Notation
                formatted_value = f"{value:.3e}"  
            else:
                formatted_value = value
            formatted_results.append((boolean, formatted_value))
        return formatted_results
    def __init__(self,R=0,F=0,amp_diff=0,T=0,Dt = 5e8,std = 0,up = 0,down = 0,mean = 0,peakA = 0, peakC = 0,lange = 0, periodicpercent = 0,StartEndDiff = 0, redshift = -1,periodicFast = 0 , classify = False):
        self.R = R
        self.F = F
        self.amp_diff = amp_diff
        self.T = T
        self.Dt = Dt
        self.std = std
        self.up = up
        self.down = down
        self.mean = mean
        self.peakA = peakA
        self.peakC = peakC
        self.lange = lange
        self.periodicpercent = periodicpercent
        self.StartEndDiff = StartEndDiff
        self.classify = classify
        self.redshift = redshift
        self.periodicFast = periodicFast
    def main(self,*, R=0,F=0,amp_diff=0,T=0,Dt = 5e8, std = 0, up = 0, down = 0,mean = 0,peakA = 0, peakC = 0,lange = 0, periodicpercent = 0, StartEndDiff = 0, redshift = -1,periodicFast = 0, classify = False):
        self.R = R
        self.F = F
        self.amp_diff = amp_diff
        self.T = T
        self.Dt = Dt
        self.std = std
        self.up = up
        self.down = down
        self.mean = mean
        self.peakA = peakA
        self.peakC = peakC
        self.lange = lange
        self.periodicpercent = periodicpercent
        self.StartEndDiff = StartEndDiff
        self.classify = classify
        self.redshift = redshift
        self.periodicFast = periodicFast
        if classify == False:
            return Conditions.periodicFast(self) and Conditions.minPoints(self) and not Conditions.supernova(self)
            #return Conditions.F_var_R(self)[0] and Conditions.minPoints(self) #(Conditions.periodic(self) or Conditions.linear(self)) and Conditions.minPoints(self) #and Conditions.minPoints(self)
        elif classify:
            return Conditions.periodic(self), Conditions.linear(self), Conditions.supernova(self), Conditions.F_var_R(self), Conditions.periodicFast(self), Conditions.minPoints(self)
    
    def F_var_R(self):
        bedingung = self.R > R_threshold and F_threshold < self.F
        #bedingung = F_threshold < self.F
        #console.print(f"R: {self.R} - {self.R > R_threshold} and F: {self.F} - {self.F > F_threshold} ---> {bedingung}")
        return bedingung, self.R+self.F
    def periodic(self):
        if self.classify == False:
            return (self.amp_diff > amp_diff_threshold and
                    self.T > T_threshold[0] and
                    self.T / (60 * 60 * 24) * 365 < self.Dt * 2)
        else:
            return (self.amp_diff > amp_diff_threshold and
                    self.T > T_threshold[0] and
                    self.T / (60 * 60 * 24) * 365 < self.Dt * 2), self.periodicpercent *100
    def periodic2(self):
        return self.amp_diff > amp_diff_threshold/2 and self.T / (60 * 60 * 24) * 365 < 365*10 and self.T /(60 * 60 * 24) * 365 > 50

    def linear(self):
        return (self.T / (60 * 60 * 24) * 365 > self.Dt * 2) and (self.amp_diff > amp_diff_threshold * 2)

    def supernova(self):
        peak = False
        width = False
        height = False
        std1 = False
        if self.peakA > 0.5 and self.peakA <= 1.6:
            peak = True
        if self.peakC > 400_000 and self.peakC <= 6_000_000:
            width = True
        if self.mean < 0.5:
            height = True
        if self.std < 0.12:
            std1 = True
        return peak and width and height and std1
    
    def changeActivity(self):
        linear = ((self.T / (60 * 60 * 24) * 365 > self.Dt * 2) and 
                self.amp_diff > amp_diff_threshold * 4)
        
        periodic = (self.amp_diff > amp_diff_threshold and 
                self.T > T_threshold[0] and 
                self.T / (60 * 60 * 24) * 365 < self.Dt * 2) and self.T / (60 * 60 * 24) * 365 > self.Dt/4 
        
        return (abs(self.StartEndDiff) > 0.25) #and (linear or periodic)  

    def periodicFast(self):
        return self.periodicFast >= 0.1
    def minPoints(self):
        return self.lange > 250
CONDITION = Conditions().main

class Plots:
    def statisticalDistribution(df):
        groups = pd.DataFrame()
        def count_true(columne):
            return sum(1 for value in columne if (value is True) or (isinstance(value, tuple) and value[0] == True))

        console.print(df.to_string())
        console.print(df.columns.to_list())

        groups = df.apply(count_true, axis=0)
        groups = groups.to_frame().T
        groups.index = ["count"]
        groups.loc['count_div_100'] = groups.loc['count'] / len(df) * 100
        console.print(len(df))
        console.print(groups)

        #console.print(groups.to_latex())
        
    def periodic_check_with_significance(name):
        curve = FileManager.load_data(name)
        #file2 = BasicCalcs.rolling_mid(curve)
        file2 = BasicCalcs.normalize_null(curve)
        file2.dropna(inplace=True)
        numeric_index = BasicCalcs.Datetime_in_Unix(file2.index)

        if len(numeric_index) < 20:
            return [], [], 0, 0, 0

        # Lomb-Scargle Periodogram
        time_diff = (numeric_index[-1] - numeric_index[0])
        frequency = np.linspace(1/(7*24*60*60*365),1/(10*24*60*60), 100000)
        
        # LombScargle mit Astropy
        ls = LombScargle(numeric_index, file2[value].values)
        power = ls.power(frequency, normalization='standard')
        fap = ls.false_alarm_probability(power.max())  # FAP für den höchsten Peak
        
        # Umrechnung der Frequenzen in Perioden
        periods = 1 / frequency
        valid_indices = periods > 0
        periods = periods[valid_indices]
        power = power[valid_indices]
        
        # Plot
        fig, (ax_t, ax_p) = plt.subplots(2, 1, constrained_layout=True)
        ax_t.plot(numeric_index, file2[value].values, 'b+')
        ax_t.set_xlabel('Time [s]')
        ax_t.set_ylabel('Amplitude')
        
        ax_p.plot(periods, power)
        #ax_p.set_xscale('log')
        ax_p.set_xlabel('Period duration [s]')
        ax_p.set_ylabel('Lomb-Scargle Power')
        ax_p.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.show()
        plt.close()

        console.print(f"False Alarm Probability (FAP): {fap:.5f}")
        if fap < 0.01:
            console.print("There is a significant periodic signal with >99% confidence!")
        elif fap < 0.05:
            console.print("There is a periodic signal with >95% confidence.")
        else:
            console.print("No significant periodic signal detected.")
    
    def periodic_check(name):
        curve = FileManager.load_data(name)
        file2 = BasicCalcs.rolling_mid(curve)
        file2 = BasicCalcs.normalize_null(curve)
        #file2 = BasicCalcs.rolling_mid(file2)
        file2.dropna(inplace=True)
        maximum = curve[value].max()
        numeric_index = BasicCalcs.Datetime_in_Unix(file2.index)
        
        # ===== FIT ======
        if len(numeric_index) < 20:
            return [],[],0,0,0
        time_diff = (numeric_index[-1] - numeric_index[0])
        w = np.linspace(1/(7*24*60*60), 1/(time_diff/4), 100000)
        
        pgram = signal.lombscargle(numeric_index, file2[value].values, w, normalize=True)
        periods = 2 * np.pi / w
        
        #====
        periods = periods/60/60/24
        numeric_index = np.array(numeric_index)/60/60/24
        #====
        
        fig, (ax_t, ax_w) = plt.subplots(2, 1, constrained_layout=True)
        ax_t.plot(numeric_index, file2[value].values, 'b+')
        ax_t.set_xlabel('Time [s]')
        ax_w.plot(periods, pgram)
        ax_w.set_xlabel('Period duration [days]')
        ax_w.set_ylabel('Normalized amplitude')
        ax_t.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax_w.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.show()

    def show_plots(sortname = pd.DataFrame()):
        galaxy_active = pd.read_csv(path+"new_active_galaxies.csv")
        files = [f for f in listdir(load_path) if isfile(join(load_path, f))]
        if len(sortname) > 0:
            name_order = {name: i for i, name in enumerate(sortname+".csv")}
            sorted_names = sorted(files, key=lambda x: name_order.get(x, float('inf')))  # Standardwert für fehlende Namen
            files = sorted_names
        
        show_galaxies_lower = [item.replace(" ","").lower()for item in show_galaxies]
        for file in tqdm(files):
            name = file[:-4].replace(" ","")
            if len(show_galaxies) > 0:
                if name.lower() not in show_galaxies_lower:
                    continue
            elif file[:-4] in galaxy_active["name"].values:
                params = FindActive.load_parameters(file[:-4])
                if not CONDITION(**params) and not config["Plots"]["IgnoreConditions"]:
                    continue
            Plots.plot_curves(file[:-4])

        
        
        
    def standart1():

        name, F_var, R, F_and_R, cuts, amplitude, amp_diff, T, periodicpercent,Dt, std, up, down, mean, peakA, peakC, lange, StartEndDiff, redshift, periodicFast = FindActive.load_parameters(variante=1)
        
        x = "F_var"
        y = "periodicFast"
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
                case "periodicpercent":
                    x = periodicpercent
                    x_title = "periodicpercent"
                    threshold_x = 0.0
                case "F_and_R":
                    x = R-(F_var*1000)
                    x_title = "F_and_R"
                    threshold_x = [F_and_R_threshold]
                case "up":
                    x = up
                    x_title = "up"
                    threshold_x = 0.0
                case "down":
                    x = down
                    x_title = "down"
                    threshold_x = 0.0
                case "std":
                    x = std
                    x_title = "std"
                    threshold_x = 0.0
                case "mean":
                    x = mean
                    x_title = "mean"
                    threshold_x = 0.0
                case "peakA":
                    x = peakA
                    x_title = "peakA"
                    threshold_x = 0.0
                case "peakC":
                    x = peakC
                    x_title = "peakC"
                    threshold_x = 0.0
                case "redshift":
                    x = redshift
                    x_title = "redshift"
                    threshold_x = 0.0
                case "periodicFast":
                    x = periodicFast
                    x_title = "periodicFast"
                    threshold_x = 0.1

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
            if CONDITION(R = R[i],F = F_var[i],amp_diff = amp_diff[i],T = T[i],Dt = Dt[i],std=std[i],up=up[i],down=down[i],mean=mean[i],peakA=peakA[i],peakC=peakC[i], lange=lange[i], periodicpercent = periodicpercent[i]):
                aktive_prozent += 1
                x_thresh.append(x[i])
                y_thresh.append(y[i])

        aktive_prozent = round(aktive_prozent / len(y) * 10000)/100
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
        error1 = file[f"{value} Error"].copy()
        file2 = BasicCalcs.rolling_mid(file,"30D")
        x1, y1 = file2.index.copy(), file2[value].copy()
        error2 = file2[f"{value} Error"].copy()

        # Kamera farben
        
        farben = config["Colors"]
        
        
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
        error_1 = file.loc[file["Filter"] == "V", f"{value} Error"].values.copy()
        error_2 = file.loc[file["Filter"] == "g", f"{value} Error"].values.copy()

        # === Plot
        galaxy_active = pd.read_csv(path + "new_active_galaxies.csv")
        # cuts = galaxy_active.loc[galaxy_active['name'] == name, 'cuts'].values[0]
        # F = galaxy_active.loc[galaxy_active["name"] == name, "activity"].values[0]
        # R = galaxy_active.loc[galaxy_active["name"] == name, "R"].values[0]
        # amp_diff = galaxy_active.loc[galaxy_active["name"] == name, "amp_diff"].values[0]
        # T = galaxy_active.loc[galaxy_active["name"] == name, "period"].values[0]
        # Dt = galaxy_active.loc[galaxy_active["name"] == name, "Dt"].values[0]
        # std = galaxy_active.loc[galaxy_active["name"] == name, "std"].values[0]
        # up = galaxy_active.loc[galaxy_active["name"] == name, "up"].values[0]
        # down = galaxy_active.loc[galaxy_active["name"] == name, "down"].values[0]
        # mean = galaxy_active.loc[galaxy_active["name"] == name, "mean"].values[0]
        # peakA = galaxy_active.loc[galaxy_active["name"] == name, "peakA"].values[0]
        # peakC = galaxy_active.loc[galaxy_active["name"] == name, "peakC"].values[0]
        # lange = galaxy_active.loc[galaxy_active["name"] == name, "pointCount"].values[0]
        # periodicpercent = galaxy_active.loc[galaxy_active["name"] == name, "periodicpercent"].values[0]
        params = FindActive.load_parameters(name)
        # try:tr_F_var = round(F*10000)/10000
        # except:tr_F_var = np.inf
        # try: tr_R = round(R*100)/100
        # except: tr_R = np.inf
        # try: tr_amplitude = round(amp_diff*100)/100
        # except: tr_amplitude = np.inf
        redshift = galaxy_active.loc[galaxy_active["name"] == name, "redshift"].values[0]
        if redshift > 0.027:
            redshift = "Neu"
        else:
            redshift = "Alt"
        if name in galaxy_active["name"].values:
            if not CONDITION(**params):
                #plt.title(f"NICHT VARIABEL Galaxy: {name}, cuts: {cuts} \nTH F: {F_threshold} Activity F: {tr_F_var}\nTR R: {R_threshold} Activity R: {tr_R}\nTR Amp: {amp_diff_threshold} Amp: {tr_amplitude}, T = {round(T/86400)} Jahre")
                plt.title(f"NICHT VARIABEL Galaxy: {name} - {redshift}")
            else:
                #plt.title(f"VARIABEL \nGalaxy: {name}, cuts: {cuts} \nTH F: {F_threshold} Activity F: {tr_F_var}\nTR R: {R_threshold} Activity R: {tr_R}\n TR Amp: {amp_diff_threshold} Amp: {tr_amplitude}, T = {round(T/86400)} Jahre")
                plt.title(f"VARIABEL \nGalaxy: {name} - {redshift}")
        else:
            plt.title(f"Galaxy: {name} - nicht gefunden")

        plt.plot(x1,y1,zorder=10, label="30 Tage", color = "red")
        plt.scatter(x_1,y_1,c = c3, alpha=0.4, zorder=5, marker = "x") # plot verschobene orginalpunkte
        plt.scatter(x_2,y_2,c = c4, alpha=0.4, zorder=5, marker = "o") # plot verschobene orginalpunkte

        #plt.errorbar(x_1, y_1, yerr=error_1, alpha=0.4, zorder=5, label='V', ecolor=c3, linestyle="None", barsabove=True,fmt="")
        #plt.errorbar(x_2, y_2, yerr=error_2, alpha=0.4, zorder=5, label="g",ecolor=c4, linestyle="None")

        x_temp,y_temp,_,_,_,_ = FindActive.periodic(name)
        #x_temp,y_temp,_,_ = FindActive.peak(name)
        plt.plot(x_temp,y_temp, label = "Sinus Fit", color = "green",zorder = 9)
        
        # ==== Standartabweichung ====
        
        #plt.hlines(y.mean()+y.std(),min(x),max(x), label = "Mittelwert", color = "black",alpha = 0.5)
        #plt.hlines(y.mean()-y.std(),min(x),max(x), label = "Mittelwert", color = "black",alpha = 0.5)

        #rolling_std = y.rolling(window="30D", center = False, min_periods = 8).std()
        #rolling_mid = y.rolling(window="30D", center = False, min_periods = 8).mean()
        #plt.plot(x,rolling_mid+rolling_std, label = "Standartabweichung", color = "black",alpha = 0.5,zorder = 6)
        #plt.plot(x,rolling_mid-rolling_std, label = "Standartabweichung", color = "black",alpha = 0.5,zorder = 6)

        
        
        # ============================
        
        plt.grid(color='grey', linestyle='--', linewidth=0.5)
        plt.xlabel("Zeit", fontsize=12)
        plt.ylabel("Fluss (normiert auf 1)", fontsize=12)
        plt.legend(fontsize=10, frameon=True, fancybox=True, framealpha=0.7)
        plt.tight_layout()
        plt.show()
        plt.close()


class FileManager:
    def loadRedshift(name1):
        name = name1.lstrip().replace(" ","").lower()
        galaxy_active = pd.read_csv("pyasassn_tool/mainTargets.csv",delimiter="|")
        galaxy_active["namecheck"] = galaxy_active["name             "].str.lstrip().str.lower().str.replace(" ", "")

        try:
            return galaxy_active.loc[galaxy_active["namecheck"] == name, "redshift"].values[0]
        except:
            return -1
    def load_data(file):
        if os.path.exists(load_path + file + ".csv"):
            data = pd.read_csv(load_path + file + ".csv", index_col = 0)
            #print(f"Test:\nIndex:\n{data.index}\nFlux:\n{data["Flux"]}\nFlux Error:\n{data["Flux Error"]}\nCamera\n{data["Camera"]}")
            data.index = pd.to_datetime(data.index)
            if len(data[value]) <= 2:
                # delete light curve
                os.remove(load_path + file + ".csv")
                # delete from active galaxies 
                activity = pd.read_csv(activity_path)
                act = activity.loc[activity["name"] != file]
                act.to_csv(activity_path, index = False)
                # delete from name_id
                new_act = pd.read_csv(config["Paths"]["path"]+"new_active_galaxies.csv")
                new_act = new_act.loc[new_act["name"] != file]
                new_act.to_csv(config["Paths"]["path"]+"new_active_galaxies.csv", index = False)
                console.log(f"Deleted: {file}")
                return -1
            return data    
        return pd.DataFrame()
    def load_cuts(name):
        activity = pd.read_csv(activity_path)
        if name in activity["name"].values:
            cuts = activity.loc[activity['name'] == name, 'cuts'].values[0]
        else: cuts = -1
        return cuts

    def group_galaxies(name):
        file_path = path+"new_active_galaxies.csv"
        cuts = FileManager.load_cuts(name)
        try:
            Fvar = FindActive.fractional_variation(name)
            if Fvar == -1:
                console.print(f"Delete: {name} (hopefully)")
                return
        except: 
            console.print(f"Delete: {name} (hopefully)")
            return
        R = FindActive.peak_to_peak_amplitudes(name)
        _,_,amplitude,amp_diff,period,periodicpercent = FindActive.periodic(name)
        _,_,peakA, peakC = FindActive.peak(name)
        deltaT = BasicCalcs.TimeDifference(name)
        std,up,down,mean,lange = FindActive.StdPeak(name)
        StartEndDiff = FindActive.changingActivity(name)
        redshift = FileManager.loadRedshift(name)
        periodicFast = FindActive.FourierFastPeriodic(name)
        if os.path.isfile(file_path) == False:
            with open(file_path, 'w') as datei:
                datei.write("name,activity,R,activity*R,cuts,amplitude,amp_diff,period,periodicpercent,Dt,std,up,down,mean,peakA,peakC,pointCount,StartEndDiff,redshift,periodicFast\n")
        with open(file_path, 'a') as datei:
            datei.write(f"{name},{Fvar},{R},{Fvar*R},{cuts},{amplitude},{amp_diff},{period},{periodicpercent},{deltaT.days},{std},{up},{down},{mean},{peakA},{peakC},{lange},{StartEndDiff},{redshift},{periodicFast}\n")
        return
    
    
class BasicCalcs:
    def derivation(x,y):
        m = []
        for i in range(len(x)-1):
            m.append((y[i+1]-y[i])/(x[i+1]-x[i]))
        
        return x[:-1], m 
    
    def format_return_values(results):
        if type(results) == type(True) or len(results) > 2:
            return results
        formatted_results = []
        for boolean, value in results:
            if isinstance(value, float):  # Überprüfen, ob es eine Zahl ist
                # Formatieren auf 3 signifikante Stellen mit wissenschaftlicher Notation
                formatted_value = f"{value:.3e}"  
            else:
                formatted_value = value
            formatted_results.append((boolean, formatted_value))
        return formatted_results
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
        maxValue = curve[value].max()
        curve[value] = curve[value]/maxValue
        curve[f"{value} Error"] = curve[f"{value} Error"]/maxValue

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
        file2[value] = file2[value].rolling(window=rolling_time, center = True, min_periods = 4).mean()
        return file2
    
    def fit_func_sin(x, a, b, c, d):
        return a * np.sin(b * x + c) + d 
    
    def fit_func_sin2(x, a, b, c, d, e, f, g, h, i, j, k, l):
        return a * np.sin(1/b * x + c) + d + e * np.sin(1/f * x + g) + h + i * np.sin(1/j * x + k) + l
    
    def fit_func_peak(x,a,b,c,d):
        return a*np.exp(-((x-b)/c)**2)+d
    
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
    def changingActivity(name,plot=False):
        curve = FileManager.load_data(name)
        curve = BasicCalcs.normalize_null(curve)
        cut = 100
        
        if plot:
            print("ZEUG: ",np.mean(curve[value][cut:]), np.mean(curve[value][:cut]) , np.mean(curve[value][cut:]) - np.mean(curve[value][:cut]))
            plt.scatter(curve.index,curve[value],color= "blue")
            plt.scatter(curve.index[:cut],curve[value][:cut],color= "red")
            plt.scatter(curve.index[-cut:],curve[value][-cut:],color= "red")
            plt.tight_layout()
            plt.show()
        if len(curve.index) > 150:
            if np.mean(curve[value][cut:]) - np.mean(curve[value][:cut]) == None:
                return 0.0
            return  np.mean(curve[value][cut:]) - np.mean(curve[value][:cut])
        else:
            return 0.0
        


    def load_parameters(name="", variante = 0):
        
        if variante == 0:
            galaxy_active = pd.read_csv(path + "new_active_galaxies.csv")
            try:
                cuts = galaxy_active.loc[galaxy_active['name'] == name, 'cuts'].values[0]
            except:
                return None
            F = galaxy_active.loc[galaxy_active["name"] == name, "activity"].values[0]
            R = galaxy_active.loc[galaxy_active["name"] == name, "R"].values[0]
            amp_diff = galaxy_active.loc[galaxy_active["name"] == name, "amp_diff"].values[0]
            T = galaxy_active.loc[galaxy_active["name"] == name, "period"].values[0]
            Dt = galaxy_active.loc[galaxy_active["name"] == name, "Dt"].values[0]
            std = galaxy_active.loc[galaxy_active["name"] == name, "std"].values[0]
            up = galaxy_active.loc[galaxy_active["name"] == name, "up"].values[0]
            down = galaxy_active.loc[galaxy_active["name"] == name, "down"].values[0]
            mean = galaxy_active.loc[galaxy_active["name"] == name, "mean"].values[0]
            peakA = galaxy_active.loc[galaxy_active["name"] == name, "peakA"].values[0]
            peakC = galaxy_active.loc[galaxy_active["name"] == name, "peakC"].values[0]
            lange = galaxy_active.loc[galaxy_active["name"] == name, "pointCount"].values[0]
            periodicpercent = galaxy_active.loc[galaxy_active["name"] == name, "periodicpercent"].values[0]
            StartEndDiff = galaxy_active.loc[galaxy_active["name"] == name, "StartEndDiff"].values[0]
            redshift = galaxy_active.loc[galaxy_active["name"] == name, "redshift"].values[0]
            periodicFast = galaxy_active.loc[galaxy_active["name"] == name, "periodicFast"].values[0]
            # liste umwandeln
            periodicFast = np.array(ast.literal_eval(periodicFast))[0]
            ehm = {
                "R": R, "F": F, "amp_diff": amp_diff, "T": T, "Dt": Dt,
                "std": std, "up": up, "down": down, "mean": mean,
                "peakA": peakA, "peakC": peakC, "lange": lange, "periodicpercent": periodicpercent,"StartEndDiff":StartEndDiff, "redshift":redshift,
                "periodicFast":periodicFast
            }
            
            return {
                "R": R, "F": F, "amp_diff": amp_diff, "T": T, "Dt": Dt,
                "std": std, "up": up, "down": down, "mean": mean,
                "peakA": peakA, "peakC": peakC, "lange": lange, "periodicpercent": periodicpercent,"StartEndDiff":StartEndDiff, "redshift":redshift,
                "periodicFast":periodicFast
            }
        elif variante == 1:
            path1 = path+"new_active_galaxies.csv"
            name = np.loadtxt(path1, delimiter=',', skiprows=1,usecols=0,dtype=str)
            F_var = np.loadtxt(path1, delimiter=',', skiprows=1,usecols=1) #F_var
            R = np.loadtxt(path1, delimiter=',', skiprows=1,usecols=2) #R
            F_and_R = np.loadtxt(path1, delimiter=',', skiprows=1,usecols=3)
            cuts = np.loadtxt(path1, delimiter=',', skiprows=1,usecols=4) #cuts
            amplitude = np.loadtxt(path1, delimiter=',', skiprows=1,usecols=5) #amplitude
            amp_diff = np.loadtxt(path1, delimiter=',', skiprows=1,usecols=6) #amplitude_diff
            T = abs(np.loadtxt(path1, delimiter=',', skiprows=1,usecols=7))
            periodicpercent = np.loadtxt(path1, delimiter=',', skiprows=1,usecols=8)
            Dt = np.loadtxt(path1, delimiter=',', skiprows=1,usecols=9) # delta T
            std = np.loadtxt(path1, delimiter=',', skiprows=1,usecols=10) # std
            up = np.loadtxt(path1, delimiter=',', skiprows=1,usecols=11) # up
            down = np.loadtxt(path1, delimiter=',', skiprows=1,usecols=12) # down
            mean = np.loadtxt(path1, delimiter=',', skiprows=1,usecols=13) # mean
            peakA = np.loadtxt(path1, delimiter=',', skiprows=1,usecols=14) # peakA
            peakC = np.loadtxt(path1, delimiter=',', skiprows=1,usecols=15) # peakC
            lange = np.loadtxt(path1, delimiter=',', skiprows=1,usecols=16) # pointCount
            StartEndDiff = np.loadtxt(path1, delimiter=',', skiprows=1,usecols=17) # StartEndDiff
            redshift = np.loadtxt(path1, delimiter=',', skiprows=1, usecols=18)  # redshift
            periodicFast = np.loadtxt(path1, delimiter=',', skiprows=1,usecols=19,dtype=str)
            print(f"Ding:{periodicFast}")
            periodicFast = np.array([float(ast.literal_eval(item)[0]) for item in periodicFast])
            #periodicFast = np.array(ast.literal_eval(periodicFast))[0]
            return name, F_var, R, F_and_R, cuts, amplitude,amp_diff, T, periodicpercent, Dt, std, up, down, mean, peakA, peakC, lange, StartEndDiff, redshift,periodicFast
            
            
    def FourierLombScargle(name,plot = False):
        curve = FileManager.load_data(name)
        #file2 = BasicCalcs.rolling_mid(curve)
        
        file2 = BasicCalcs.normalize_null(curve)
        file2.dropna(inplace=True)
        file2[value] = file2[value] - 0.5
        numeric_index = BasicCalcs.Datetime_in_Unix(file2.index)

        if len(numeric_index) < 20:
            return [], [], 0, 0, 0

        # Lomb-Scargle Periodogram

        frequency, power = LombScargle(numeric_index, file2[value]).autopower(minimum_frequency=1e-9,maximum_frequency = 1e-6, samples_per_peak = 20)
        peaks, properties = signal.find_peaks(power, height=0.003)
        sorted_indices = np.argsort(properties['peak_heights'])[::-1]
        sorted_peaks = peaks[sorted_indices]
        sorted_properties = properties['peak_heights'][sorted_indices]

        peaks = sorted_peaks[0:3]
        if plot: 
            if True:
                teilen = (60*60*24*365)
            else:
                teilen = 1
            if max(power) >= 0.003:
                pass
                #power = power / max(power)
            #console.print(f"PEAKS: {frequency[peaks]*(2*np.pi)}")
        
            x,y,Tsin = FindActive.periodic(name,plot = True)
            console.print(f"Periods: {1/(frequency[peaks])}, total - {max(x)-min(x)}")
            x = np.array(x)/teilen
            x = x-min(x)
            numeric_index = np.array(numeric_index)/teilen
            numeric_index = numeric_index-min(numeric_index)
            fig, (ax_t, ax_w) = plt.subplots(2, 1, constrained_layout=True)
            plt.title(f"Galaxy: {name}")
            ax_t.plot(numeric_index, file2[value].values, 'b+')
            ax_t.plot(x,y-0.5, label = "Sinus Fit", color = "green")
            ax_t.set_xlabel(f'Zeit in [Jahren]\nTime [1/y] - Fit: {Tsin/(2*np.pi)*teilen} "Fourier: {frequency[peaks]*teilen} -> Fit T = {1/(Tsin/(2*np.pi)*teilen)}')
            ax_w.plot((frequency*teilen), power)
            ax_w.vlines((Tsin/(2*np.pi)*teilen),min(power),max(power),color = "red")
            ax_w.set_xlabel('Period duration [1/Jahre]')
            ax_w.set_ylabel('Normalized amplitude')
            ax_t.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax_w.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.get_current_fig_manager().full_screen_toggle()
            plt.show()
        return 2*np.pi*frequency[peaks], power[peaks]

    def FourierFastPeriodic(name):
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

        file = pd.read_csv(f"final_light_curves/{name}.csv")
        file = normalize(file)

        t = Datetime_in_Unix(pd.to_datetime(file["JD"]))
        mint = min(t)
        for i in range(len(t)):
            t[i] -= mint
        y = []
        for val in file["Flux"].values:
            y.append(val)

        frequency, power = LombScargle(t, y).autopower(maximum_frequency=1e-7, samples_per_peak=40)
        freq = pd.DataFrame({"Frequency": frequency, "Power": power})
        return [max(freq.loc[freq["Frequency"]>= 1e-8, "Power"].values)]



    def peak_to_peak_amplitudes(name): # returns R
        curve = FileManager.load_data(name)
        #file2 = BasicCalcs.normalize(curve)
        file2 = curve.copy()
        #print(f"Max: {file2[value].max()} Min: {file2[value].min()}, Zsm: {file2[value].max() / file2[value].min()}")
        if file2[value].max() / file2[value].min() <= 0:
            console.log(f"\nALARM {file2[value].max() / file2[value].min()}\n")
        return file2[value].max() / file2[value].min()
    
    def fractional_variation(name):
        curve = FileManager.load_data(name)
        try:
            if curve == -1:
                return -1
        except: 
            pass
        file2 = curve.copy()
        file2 = BasicCalcs.normalize(file2)
        activity = (file2[value].std()**2-BasicCalcs.delta(file2)**2) / file2[value].mean()
        return activity
    
    def peak(name):
        curve = FileManager.load_data(name)
        file2 = BasicCalcs.normalize_null(curve)
        #file2 = BasicCalcs.rolling_mid(file2)
        file2.dropna(inplace=True)
        maximum = curve[value].max()
        numeric_index = BasicCalcs.Datetime_in_Unix(file2.index)
        
        # ===== FIT =====
        if len(numeric_index) < 2:
            return [],[],0,0
        time_diff = (numeric_index[-1] - numeric_index[0]) 
        peak_index = file2[value].idxmax()
        peak_time = BasicCalcs.Datetime_in_Unix([peak_index])[0]
        amp = curve[value].max() - curve[value].min()
        params, params_covariance = optimize.curve_fit(BasicCalcs.fit_func_peak, numeric_index, file2[value].values, p0=[1.2,peak_time , 10000000, 0.0],maxfev=100000, bounds=([-1,0.8*numeric_index[0],1000,0],[4,1.2*numeric_index[-1],10000000,0.5])) # a*e^-((x+b)/c)^2 + d
        x = np.linspace(min(numeric_index), max(numeric_index), 10000)
        y = BasicCalcs.fit_func_peak(x, *params) 
        x = BasicCalcs.Unix_in_Datetime(x)
        if len(params) < 4:
            print("eh")
            return x,y,0,0
        return x,y,params[0],params[2] 
        
    
    def periodic(name, plot = False):
        curve = FileManager.load_data(name)
        file2 = BasicCalcs.rolling_mid(curve)
        file2 = BasicCalcs.normalize_null(curve)
        #file2 = BasicCalcs.rolling_mid(file2)
        file2.dropna(inplace=True)
        maximum = curve[value].max()
        numeric_index = BasicCalcs.Datetime_in_Unix(file2.index)
        
        # ===== FIT ======
        if len(numeric_index) < 20:
            return [],[],0,0,0,0
        time_diff = (numeric_index[-1] - numeric_index[0])         
        amp = curve[value].max() - curve[value].min()

        try:
            params, params_covariance = optimize.curve_fit(BasicCalcs.fit_func_sin, numeric_index, file2[value].values, p0=[amp, 1/time_diff, 0, file2[value].mean()],maxfev=100000) # a * np.sin(b * x + c) + d 
        except:
            return [],[],0,0,0,0
        x = np.linspace(min(numeric_index), max(numeric_index), 10000)
        y = BasicCalcs.fit_func_sin(x, *params)
        
        Tsin = abs(params[1])
        if plot:
            return x,y,Tsin
        
        x = BasicCalcs.Unix_in_Datetime(x)
        T,peaks = FindActive.FourierLombScargle(name)
        #console.print(f"Tsin: {Tsin}, T: {T}, Peaks: {peaks}")
        for i in range(len(T)):
            if T[i]/Tsin < 1.2 and T[i]/Tsin > 0.8:
                #console.print(f"Period T: {T[i]} Tsin: {Tsin} T/Tsin: {T[i]/Tsin}")
                return x,y, abs(params[0]), (y.max() - y.min()),(1/abs(params[1]*60)),peaks[i]
            else:
                pass
                #console.print(f"Passt nicht {i} Period T: {T[i]} Tsin: {Tsin} T/Tsin: {T[i]/Tsin}")



        
        
        return x,y, abs(params[0]), (y.max() - y.min()),(1/abs(params[1]*60)),0 # T in sekunden       
    
    def periodic3(name):
        # =====
        
        #Klappt nicht so schön
        
        # =====
        curve = FileManager.load_data(name)
        file2 = BasicCalcs.rolling_mid(curve)
        file2 = BasicCalcs.normalize_null(curve)
        #file2 = BasicCalcs.rolling_mid(file2)
        file2.dropna(inplace=True)
        maximum = curve[value].max()
        numeric_index = BasicCalcs.Datetime_in_Unix(file2.index)
        
        # ===== FIT ======
        if len(numeric_index) < 20:
            return [],[],0,0,0
        time_diff = (numeric_index[-1] - numeric_index[0]) 
        
        amp = curve[value].max() - curve[value].min()
        try:
            params, params_covariance = optimize.curve_fit(BasicCalcs.fit_func_sin2, numeric_index, file2[value].values, p0=[500, 1e12, 0, file2[value].mean(),20, 1e9, 0, file2[value].mean(),0.5, 1e7, 0, file2[value].mean()],maxfev=100000,
                                                        bounds=([-10_000,1e8,-4,-5_000,
                                                                 -100,1e6,-4,-100,
                                                                 -2,1e4,-4,-2],
                                                                [10_000,1e15,4,5_000,
                                                                 100,1e10,4,100,
                                                                 2,1e8,4,2]))
            a,e,i,b,f,j  = params[0],params[4],params[8],params[1],params[5],params[9]
            console.print(f"Params: {a,e,i,b,f,j}")
        except:
            console.log("No Parameters found")
            return [],[],0,0,0
        x = np.linspace(min(numeric_index), max(numeric_index), 10000)
        y = BasicCalcs.fit_func_sin2(x, *params)
        x = BasicCalcs.Unix_in_Datetime(x)
        return x,y, abs(params[0]), (y.max() - y.min()),(1/abs(params[1]*60)) # T in sekunden 
    
    def StdPeak(name):
        curve = FileManager.load_data(name)
        lange = len(curve)
        curve = BasicCalcs.normalize_null(curve)
        
        mean = curve[value].mean()
        std = curve[value].std()
        rolling = curve[value].rolling(window="30D", center = False, min_periods = 8)
        rolling_std = rolling.std().values
        rolling_mid = rolling.mean().values
        
        up = 0
        down = 0
        for i in range(len(rolling_std)):
            if rolling_mid[i] + rolling_std[i] > mean + std:
                up += 1
            elif rolling_mid[i] - rolling_std[i] < mean - std:
                down += 1
                
        return std, up, down, mean, lange
    def jumps(name):
        curve = FileManager.load_data(name)
        file2 = BasicCalcs.normalize(curve)
               
        
        
    def stetig(name):
        curve = FileManager.load_data(name)
        file2 = BasicCalcs.normalize(curve)
        

    
def start():
    if not config["ReCalculateOnlyNew"] or not os.path.exists(path+"new_active_galaxies.csv"):
        with open(path+"new_active_galaxies.csv", 'w') as datei:
            datei.write("name,activity,R,activity*R,cuts,amplitude,amp_diff,period,periodicpercent,Dt,std,up,down,mean,peakA,peakC,pointCount,StartEndDiff,redshift,periodicFast\n")
    galaxy_active = pd.read_csv(path+"new_active_galaxies.csv") # wurden bereits berechnet
    files = [f for f in listdir(load_path) if isfile(join(load_path, f))]
    show_galaxies_lower = [item.replace(" ","").lower()for item in show_galaxies]


    if config["ReCalculateOnlyNew"]:
        savedCurves = [f[:-4] for f in files]
        for i in reversed(range(len(savedCurves))):
            if savedCurves[i] in galaxy_active["name"].values:
                savedCurves.pop(i)
                files.pop(i)

    for file in tqdm(files):
        #if file[:-4] not in galaxy_active["name"].values or not config["ReCalculateOnlyNew"]:
        FileManager.group_galaxies(file[:-4])
        #else: continue
        

if config["ReCalculate"] or config["ReCalculateOnlyNew"]:
    start()
if config["Plots"]["ShowAllPlots"]: 
    Plots.show_plots()
if config["Plots"]["ShowGroupPlot"]:
    Plots.standart1()   
if config["Plots"]["ShowFourierPlot"]:
    console.print(groups)
    if show_galaxies != []:
        for name in show_galaxies:
            params = FindActive.load_parameters(name)
            if CONDITION(**params,classify=True) or config["Plots"]["IgnoreConditions"]:
                FindActive.FourierLombScargle(name, plot = True)
    else:
        liste = listdir("final_light_curves")
        for i in liste:
            params = FindActive.load_parameters(i[:-4])
            if CONDITION(**params,classify=True) or config["Plots"]["IgnoreConditions"]:
                FindActive.FourierLombScargle(i[:-4],plot = True)
if config["Plots"]["ShowClassifyPlot"] or config["Plots"]["sortedPlot"] or config["Plots"]["statisticalDistribution"]:
    liste = listdir("final_light_curves")
    galaxy_active = pd.read_csv(path + "new_active_galaxies.csv")
    groups = pd.DataFrame()
    for file in liste:
        name = file[:-4]
        if name not in show_galaxies and show_galaxies != []:
            continue
        params = FindActive.load_parameters(name)
        try:
            add = pd.DataFrame([list(CONDITION(**params,classify=True))],index = [name],columns=["periodic","linear","supernova","F_var_R","periodicFast","minPoint"])
            groups = pd.concat([groups, add])
        except:
            continue
    def count_true(row):
        return sum(1 for value in row if (value is True) or (isinstance(value, tuple) and value[0] == True))

    groups["True_Count"] = groups.apply(count_true, axis=1)
    df_sorted = groups.sort_values(by="True_Count", ascending=False)
    df_sorted = df_sorted[df_sorted["minPoint"] == True]
    #df_sorted = df_sorted.drop(columns=["True_Count"])
    
    if config["Plots"]["statisticalDistribution"]:
        Plots.statisticalDistribution(df_sorted)
    if config["Plots"]["ShowClassifyPlot"]:    
        console.print(df_sorted.to_string())
    if config["Plots"]["sortedPlot"]:
        #df_sorted.to_csv("sortedcurves.csv")
        Plots.show_plots(df_sorted.index)
            
if config["Plots"]["changeActivity"]:
    if show_galaxies != []:
        for name in show_galaxies:
            params = FindActive.load_parameters(name)
            if CONDITION(**params,classify=True) or config["Plots"]["changeActivity"]:
                FindActive.changingActivity(name, plot = True)
    else:
        liste = listdir("final_light_curves")
        for i in liste:
            params = FindActive.load_parameters(i[:-4])
            if CONDITION(**params,classify=True) or config["Plots"]["changeActivity"]:
                FindActive.changingActivity(i[:-4],plot = True)
   
   
"""

    TODO: beim speichern der finalen kurve leerzeichen entfernen
    TODO: redshift in finalen kurven speichern 
    
"""



# NGC 676, NGC 1275, NGC 3516,NGC 5273,NGC 4253,NGC 5548,Mrk 1044

#PG 1149-110
#NGC 4235