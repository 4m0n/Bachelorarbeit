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
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize


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
    def __init__(self,R=0,F=0,amp_diff=0,T=0,Dt = 5e8,std = 0,up = 0,down = 0,mean = 0,peakA = 0, peakC = 0,lange = 0, periodicpercent = 0,StartEndDiff = 0, redshift = -1,periodicFast = 0, magnitude = 0, periodicFastFreq = 0, amplitude = 0, classify = False):
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
        self.magnitude = magnitude
        self.periodicFastFreq = periodicFastFreq
        self.amplitude = amplitude
    def main(self,*, R=0,F=0,amp_diff=0,T=0,Dt = 5e8, std = 0, up = 0, down = 0,mean = 0,peakA = 0, peakC = 0,lange = 0, periodicpercent = 0, StartEndDiff = 0, redshift = -1,periodicFast = 0, magnitude = 0, periodicFastFreq = 0 ,amplitude = 0, classify = False):
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
        self.magnitude = magnitude
        self.periodicFastFreq = periodicFastFreq
        self.amplitude = amplitude
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
        cond = (self.amp_diff > amp_diff_threshold and
                    self.T > T_threshold[0] and
                    self.T / (60 * 60 * 24) * 365 < self.Dt * 2) and self.T > 1e8
        cond =  self.amplitude >= 0.15
        if self.classify == False:
            return cond
        else:
            return cond, self.periodicpercent * 100
    def periodic2(self):
        return self.amp_diff > amp_diff_threshold/2 and self.T / (60 * 60 * 24) * 365 < 365*10 and self.T /(60 * 60 * 24) * 365 > 50

    def linear(self):
        return (1/(self.T) / (60 * 60 * 24) * 365 > self.Dt * 2) and (self.amplitude > 0.5)

    def supernova(self):
        peak = False
        width = False
        height = False
        std1 = False
        if self.peakA > 0.25 and self.peakA <= 3:
            peak = True
        if self.peakC > 400_000 and self.peakC <= 10_000_000:
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
    def printAllParameters(self):
        string = f"R: {self.R}, F: {self.F}, amp_diff: {self.amp_diff}, T: {self.T}, Dt: {self.Dt}, std: {self.std}, up: {self.up}, down: {self.down}, mean: {self.mean}, peakA: {self.peakA}, peakC: {self.peakC}, lange: {self.lange}, periodicpercent: {self.periodicpercent}, StartEndDiff: {self.StartEndDiff}, classify: {self.classify}, redshift: {self.redshift}, periodicFast: {self.periodicFast}, magnitude: {self.magnitude}, periodicFastFreq: {self.periodicFastFreq}, amplitude: {self.amplitude}"
        return string
CONDITION = Conditions().main

class Plots:

    @staticmethod
    def plotMateches(df,df_names):
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df, annot=True, cmap='Blues', linewidths=.5, linecolor='black',
                    cbar_kws={'label': 'Correlation'})
        plt.title('Heatmap of Values', fontsize=18)
        plt.xlabel('berechnet', fontsize=14)
        plt.ylabel('manuell', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)

        def on_click(event):
            if event.inaxes == ax:  # Check if the click is inside the heatmap
                x, y = int(event.xdata), int(event.ydata)  # Round to nearest index
                if 0 <= x < len(normalized_df.columns) and 0 <= y < len(normalized_df.index):
                    x_label = normalized_df.columns[x]
                    y_label = normalized_df.index[y]

                    # Retrieve the corresponding data
                    SN = df_names[x_label][y_label]
                    plot = SN

                    print(f"\n\n======== PLOTS len:{len(plot)}========\nn")
                    print(f"\nx = {x_label} ===== y = {y_label}\n")
                    # Generate a new plot for each value in `plot`
                    for val in plot:
                        fig_new, ax_new = plt.subplots()
                        params = FindActive.load_parameters(val)
                        print(f"=== {val} ===\n{Conditions(**params).printAllParameters()}")
                        Plots.plot_curves(val,True)
                        plt.show(block=False)
                        plt.pause(0.1)
                        if input("nächster Plot: (break for exit) ") == "break":
                            print("exit")
                            plt.close()
                            break

                        plt.close()

        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()

        plt.close()
    @staticmethod
    def matchesPlot(df):
        df = df.drop(columns=["name"])
        df.set_index("category", inplace=True)

        category_counts = df.groupby("category")["match"].value_counts().unstack(fill_value=0)
        category_counts.drop(columns=["match: False - Notes: Leer, Calc: Leer"], inplace=True)
        category_counts = category_counts.drop(index=False)
        category_counts["corr"] = category_counts[True]/(category_counts[False] + category_counts[True])

        heatmap_data = pd.pivot_table(category_counts, values="corr", index="category", columns="category", fill_value=0)

        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, cmap='Blues', linewidths=.5, linecolor='black', cbar_kws={'label': 'Correlation'})
        plt.title('Heatmap of Values', fontsize=18)
        plt.xlabel('X Name', fontsize=14)
        plt.ylabel('Y Name', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.show()
        plt.close()



    def statisticalDistribution(df):
        groups = pd.DataFrame()
        def count_true(columne):
            return sum(1 for value in columne if (value is True) or (isinstance(value, tuple) and value[0] == True))

        console.print(df.to_string())
        console.print(df.columns.to_list())

        groups = df.apply(count_true, axis=0)
        groups = groups.to_frame().T
        groups.index = ["count"]
        groups.loc['percent'] = groups.loc['count'] / len(df) * 100
        console.print(len(df))
        console.print(groups)
        df = df.drop(columns=["minPoint","True_Count"])

        print(df)
        rows_with_true = df.map(lambda x: x[0] if isinstance(x, tuple) else x).any(axis=1)
        count = rows_with_true.sum()
        print(f"Variable Insgesamt: {count}, percent: {groups.loc["count","total"]/count}")


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

        
        
        
    def standart1(x=None,y=None,z=None,plot = True):

        name, F_var, R, F_and_R, cuts, amplitude, amp_diff, T, periodicpercent, Dt, std, up, down, mean, peakA, peakC, lange, StartEndDiff, redshift, periodicFast, magnitude, periodicFastFreq = FindActive.load_parameters(variante=1)
        if x == None:
            x = "F_and_R"
        if y == None:
            y = "periodicFast"
        if z == None:
            z, z_title = cuts, "cuts"
        else:
            z_title = "Noch benennen"
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
                case "F_and_R":
                    x = R-(F_var*1000)
                    x_title = "F_and_R"
                    threshold_x = [F_and_R_threshold]
                case "cuts":
                    x = cuts
                    x_title = "cuts"
                    threshold_x = 0.0
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
                case "Dt":
                    x = Dt
                    x_title = "Dt"
                    threshold_x = 0.0
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
                case "lange":
                    x = lange
                    x_title = "länge"
                    threshold_x = 0.0
                case "StartEndDiff":
                    x = StartEndDiff
                    x_title = "StartEndDiff"
                    threshold_x = 0.0
                case "redshift":
                    x = redshift
                    x_title = "redshift"
                    threshold_x = 0.0
                case "periodicFast":
                    x = periodicFast
                    x_title = "periodicFast"
                    threshold_x = 0.1
                case "magnitude":
                    x = magnitude
                    x_title = "magnitude"
                    threshold_x = 0.1
                case "periodicFastFreq":
                    x = periodicFastFreq
                    x_title = "periodicFastFreq"
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
        corrData = pd.DataFrame(columns=["x","y","z"])
        corrData["x"] = x
        corrData["y"] = y
        corrData["z"] = z
        # cuts repräsentieren je
        if x_title == "magnitude":
            corrData["x"] = abs(corrData["x"])
        if y_title == "magnitude":
            corrData["y"] = abs(corrData["y"])
        if x_title == "cuts":
            corrData = corrData[(corrData["x"] >= 0) & (corrData["y"] >= 0)]
        elif y_title == "cuts":
            corrData = corrData[(corrData["x"] >= 0) & (corrData["y"] >= 0)]
        else:
            try:
                corrData = corrData[(corrData["x"] > 0) & (corrData["y"] > 0)]
            except:
                for val in corrData["x"]:
                    print(f"{val} - {type(val)}")
                for val in corrData["y"]:
                    print(f"{val} - {type(val)}")
                print(f"datax: {corrData['x']}\n datay: {corrData['y']}")
                print(f"x: {x_title} y: {y_title}")
        corrData['z_x'] = (corrData['x'] - corrData['x'].mean()) / corrData['x'].std()
        corrData['z_y'] = (corrData['y'] - corrData['y'].mean()) / corrData['y'].std()
        threshold = 3
        # Ausreisser entfernen für zuverlässigere Ergebnisse
        corrData = corrData[(corrData['z_x'].abs() < threshold) & (corrData['z_y'].abs() < threshold)]
        corrData = corrData.drop(columns=['z_x', 'z_y'])
        lenbefore = len(x)
        x,y,z = corrData["x"], corrData["y"],corrData["z"]
        lenafter = len(x)
        geloscht = lenbefore - lenafter
        corr = np.corrcoef(x, y)[0, 1]

        if plot:
            if x_title == "magnitude":
                x*=-1
            if y_title == "magnitude":
                y*=-1
            print(f"Correlation: {corr} - bereinigt um {geloscht} Datenpunkte")
            if type(threshold_x) == type(4.20):
                threshold_x = [threshold_x]
            if type(threshold_y) == type(4.20):
                threshold_y = [threshold_y]
            for i in threshold_y:
                continue
                plt.hlines(i,min(x),max(x)) # R
            for i in threshold_x:
                continue
                plt.vlines(i,min(y),max(y)) # F
            plt.scatter(x,y,c = z,s = 50, edgecolor = "k", alpha = 0.5,zorder = 1)
            #plt.scatter(x_target,y_target,c = "red",s = 50, edgecolor = "k", alpha = 0.8,marker = 5,zorder = 2)
            #plt.scatter(x_thresh,y_thresh,c = "blue",s = 50, edgecolor = "k", alpha = 0.5,marker = 4,zorder = 1)

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
            plt.title(f"Anzahl: {len(x)}, Variabel: {aktive_prozent}%\nFilter trifft {prozent_korrekte_vorhersagen}% der Galaxien\nKorrekt in Filter: {prozent_treffer_in_filter}%\nCorrelation: {corr}")
            plt.grid()
            plt.show()
            plt.close()
        return corr,geloscht,len(x)

    def plot_curves(name,deactivePlot=False):

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
        redshift = galaxy_active.loc[galaxy_active["name"] == name, "redshift"].values
        if redshift > 0.027:
            redshift = "Neu"
        else:
            redshift = "Alt"

        compareString,_,_ = FindActive.compareCategories(name)
        if name in galaxy_active["name"].values:
            if not CONDITION(**params):
                #plt.title(f"NICHT VARIABEL Galaxy: {name}, cuts: {cuts} \nTH F: {F_threshold} Activity F: {tr_F_var}\nTR R: {R_threshold} Activity R: {tr_R}\nTR Amp: {amp_diff_threshold} Amp: {tr_amplitude}, T = {round(T/86400)} Jahre")
                plt.title(f"NICHT VARIABEL Galaxy: {name} - {redshift}\n{compareString}")
            else:
                #plt.title(f"VARIABEL \nGalaxy: {name}, cuts: {cuts} \nTH F: {F_threshold} Activity F: {tr_F_var}\nTR R: {R_threshold} Activity R: {tr_R}\n TR Amp: {amp_diff_threshold} Amp: {tr_amplitude}, T = {round(T/86400)} Jahre")
                plt.title(f"VARIABEL \nGalaxy: {name} - {redshift}\n{compareString}")
        else:
            plt.title(f"Galaxy: {name} - nicht gefunden")


        plt.plot(x1,y1,zorder=10, label="30 Tage", color = "red")
        plt.scatter(x_1,y_1,c = c3, alpha=0.4, zorder=5, marker = "x") # plot verschobene orginalpunkte
        plt.scatter(x_2,y_2,c = c4, alpha=0.4, zorder=5, marker = "o") # plot verschobene orginalpunkte

        #plt.errorbar(x_1, y_1, yerr=error_1, alpha=0.4, zorder=5, label='V', ecolor=c3, linestyle="None", barsabove=True,fmt="")
        #plt.errorbar(x_2, y_2, yerr=error_2, alpha=0.4, zorder=5, label="g",ecolor=c4, linestyle="None")

        #x_temp,y_temp,_,_,_,_ = FindActive.periodic(name)
        #x_temp,y_temp,_,_ = FindActive.peak(name)
        #plt.plot(x_temp,y_temp, label = "Sinus Fit", color = "green",zorder = 9)
        
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
        if deactivePlot == False:
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
        periodicFast,periodicFastFreq = FindActive.FourierFastPeriodic(name)
        magnitude = FindActive.absoluteMag(name)
        if os.path.isfile(file_path) == False:
            with open(file_path, 'w') as datei:
                datei.write("name,activity,R,activity*R,cuts,amplitude,amp_diff,period,periodicpercent,Dt,std,up,down,mean,peakA,peakC,pointCount,StartEndDiff,redshift,periodicFast,periodicFastFreq,magnitude\n")
        with open(file_path, 'a') as datei:
            datei.write(f"{name},{Fvar},{R},{Fvar*R},{cuts},{amplitude},{amp_diff},{period},{periodicpercent},{deltaT.days},{std},{up},{down},{mean},{peakA},{peakC},{lange},{StartEndDiff},{redshift},{periodicFast},{periodicFastFreq},{magnitude}\n")
        return
    
    
class BasicCalcs:
    @staticmethod
    def secondToYears(value):
        return value/(365*24*60*60)
    @staticmethod
    def yearsToSeconds(value):
        return value*(365*24*60*60)
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
    @staticmethod
    def GetCategroies():
        files = pd.read_csv("sortedcurves.csv")
        matches = pd.DataFrame()
        noteskategorien = {
            1: "inactive",
            2: "leichte var",
            3: "mittlere var",
            4: "Anstieg/Abfall",
            5: "starke var",
            6: "Periode >2J",
            7: "Periode 0.5-2J",
            8: "Spezialfälle",
            9: "SN",
        }
        noteskategorienCalc = {
            0: "periodic",
            1: "linear",
            2: "supernova",
            3: "F_var_R",
            4: "periodicFast",
            5: "inactive"
        }
        df = pd.DataFrame(0, index=noteskategorienCalc.values(), columns=noteskategorien.values())
        df_names = pd.DataFrame(index=noteskategorienCalc.values(), columns=noteskategorien.values(), dtype=object)

        for col in df_names.columns:
            for row in df_names.index:
                df_names.at[row, col] = []

        check = 0
        check1 = 0
        galNotes = pd.read_csv("Lichtkurven/galaxienotes.csv") # notes
        galCalc = pd.read_csv("sortedcurves.csv") # algorithm
        for val in files["Unnamed: 0"].values:
            check1 += 1
            galCalc.columns = ["name"] + list(galCalc.columns[1:])  # Add "name" as the first column header
            galCalc = galCalc.rename_axis("name", axis="index")

            galNotes["name"] = galNotes["name"].str.lower().str.replace(" ", "")
            galCalc["name"] = galCalc["name"].str.lower().str.replace(" ", "")
            nameCompare = val.lower().replace(" ", "")

            catNotes = galNotes.loc[galNotes["name"] == nameCompare,"category"].values
            catCalc = galCalc.loc[galCalc["name"] == nameCompare].values
            try:
                notesCat = noteskategorien[catNotes[0]]
            except:
                notesCat = "inactive"



            calcTrue = []
            calcTrueNames = ""
            try:
                catCalc = catCalc[0]
            except:
                catCalc = ['name','(False, 0.0)',False,False,'(False, 0)',False,False,1]

            inactive = True
            for i in range(len(catCalc)-2):
                if i == 0:
                    continue
                if (isinstance(catCalc[i], str) and "True" in catCalc[i]) or (isinstance(catCalc[i], bool) and True == catCalc[i]):
                    calcTrue.append(noteskategorienCalc[i-1])
                    calcTrueNames = val
                    inactive = False
            if inactive:
                calcTrue.append("inactive")
                calcTrueNames = val

            for value in calcTrue:
                df.loc[value, notesCat] += 1
                df_names.loc[value, notesCat].append(calcTrueNames)
                check += 1


        print(df_names)
        # notesCat: F_var_R
        # calcTrue: [F_var_r,linear]
        print(df)

        # Calculate the total sum
        total_sum = df.values.sum()

        print(f"Total sum:{total_sum} insgesamt:{check} - obere schleife: {check1}")
        return df,df_names

    @staticmethod
    def MatchCategory(notes, calc):
        if len(notes) == 0 or len(calc) == 0:
            print("ehm")
            return f"match: False - Notes: Leer, Calc: Leer", False,False
        calc = calc[0]
        noteskategorienCalc = {
            0: "periodic",
            1: "linear",
            2: "supernova",
            3: "F_var_R",
            4: "periodicFast",
        }
        noteskategorien = {
            1: "inactive",
            2: "leichte var",
            3: "mittlere var",
            4: "Anstieg/Abfall",
            5: "starke var",
            6: "Periode >2J",
            7: "Periode 0.5-2J",
            8: "Spezialfälle",
            9: "SN",
        }
        calcTrue = []
        for i in range(len(calc)-2):
            if i == 0:
                continue
            if (isinstance(calc[i], str) and "True" in calc[i]) or (isinstance(calc[i], bool) and True == calc[i]):
                calcTrue.append(noteskategorienCalc[i-1])
        print(f"notes: {notes}")
        print(notes[0])
        if notes[0] == 0:
            notes[0] =1
        notesTrue = noteskategorien[notes[0]]
        match = False
        if notesTrue in calcTrue:
            match = True
        return match,notesTrue,calcTrue
    @staticmethod
    def compareCategories(name):
        galNotes = pd.read_csv("Lichtkurven/galaxienotes.csv")
        galCalc = pd.read_csv("sortedcurves.csv")
        galCalc.columns = ["name"] + list(galCalc.columns[1:])  # Add "name" as the first column header
        galCalc = galCalc.rename_axis("name", axis="index")

        galNotes["name"] = galNotes["name"].str.lower().str.replace(" ", "")
        galCalc["name"] = galCalc["name"].str.lower().str.replace(" ", "")


        nameCompare = name.lower().replace(" ", "")
        catNotes = galNotes.loc[galNotes["name"] == nameCompare,"category"].values
        catCalc = galCalc.loc[galCalc["name"] == nameCompare].values

        match,notesTrue,calcTrue = FindActive.MatchCategory(catNotes, catCalc)
        compareString = f"from Notes: {notesTrue}\nCalculated: {', '.join(calcTrue)}"

        return compareString, match,notesTrue

    def absoluteMag(name):
        name = name.replace(" ","").lower()
        galaxy_active = pd.read_csv("pyasassn_tool/mainTargets.csv",delimiter="|")
        galaxy_active["namecheck"] = galaxy_active["name             "].str.lstrip().str.lower().str.replace(" ", "")
        try:
            mag = galaxy_active.loc[galaxy_active["namecheck"] == name, "abs_mag"].values[0]
            if type(mag) == str:
                mag = float(mag.replace(" ",""))
        except:
            mag = 0

        return mag

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
            galaxy_active = pd.read_csv(path + "new_active_galaxies.csv",delimiter=",")
            try:
                cuts = galaxy_active.loc[galaxy_active['name'] == name, 'cuts'].values[0]
            except:
                return None
            F = galaxy_active.loc[galaxy_active["name"] == name, "activity"].values[0]
            R = galaxy_active.loc[galaxy_active["name"] == name, "R"].values[0]
            amp_diff = galaxy_active.loc[galaxy_active["name"] == name, "amp_diff"].values[0]
            amplitude = galaxy_active.loc[galaxy_active["name"] == name, "amplitude"].values[0]
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
            periodicFast = float(galaxy_active.loc[galaxy_active["name"] == name, "periodicFast"].values[0])
            periodicFastFreq = galaxy_active.loc[galaxy_active["name"] == name, "periodicFastFreq"].values[0]
            magnitude = float(galaxy_active.loc[galaxy_active["name"] == name, "magnitude"].values[0])
            # liste umwandeln

            return {
                "R": R, "F": F, "amp_diff": amp_diff, "T": T, "Dt": Dt,
                "std": std, "up": up, "down": down, "mean": mean,
                "peakA": peakA, "peakC": peakC, "lange": lange, "periodicpercent": periodicpercent,"StartEndDiff":StartEndDiff, "redshift":redshift,
                "periodicFast":periodicFast,"magnitude":magnitude,"periodicFastFreq":periodicFastFreq,"amplitude":amplitude
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
            periodicFast = np.loadtxt(path1, delimiter=',', skiprows=1,usecols=19,dtype=float)
            periodicFastFreq = np.loadtxt(path1, delimiter=',', skiprows=1, usecols=20, dtype=float)
            magnitude = np.loadtxt(path1, delimiter=',', skiprows=1,usecols=21,dtype=float)
            #periodicFast = np.array(ast.literal_eval(periodicFast))[0]
            return name, F_var, R, F_and_R, cuts, amplitude,amp_diff, T, periodicpercent, Dt, std, up, down, mean, peakA, peakC, lange, StartEndDiff, redshift,periodicFast,magnitude,periodicFastFreq
            
            
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

        # Frequenzbereich für 3,17 bis 50 Jahre:
        min_freq = 1/(2*365.25 *24*60*60) #1 / (2 * 365.25 * 24 * 3600)  # 2 jahre
        max_freq = 1 / (0.08 * 365.25 * 24 * 3600)  # ~100 tage

        frequency, power = LombScargle(t, y).autopower(
            minimum_frequency=min_freq,
            maximum_frequency=max_freq,
            samples_per_peak=40
        )
        max_power_idx = np.argmax(power)
        dominant_freq = frequency[max_power_idx]
        dominant_power = power[max_power_idx]
        return dominant_power,dominant_freq  # Maximaler Power-Wert im gewählten Bereich



    def peak_to_peak_amplitudes(name): # returns R
        curve = FileManager.load_data(name)
        #file2 = BasicCalcs.normalize(curve)
        file2 = curve.copy()
        #print(f"Max: {file2[value].max()} Min: {file2[value].min()}, Zsm: {file2[value].max() / file2[value].min()}")
        if file2[value].max() / file2[value].min() <= 0:
            console.log(f"\nALARM {name}  -  {file2[value].max() / file2[value].min()}\n")
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

    def periodic(name, plot=False):
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
        curve = file.copy()
        file = normalize(file)

        t = Datetime_in_Unix(pd.to_datetime(file["JD"]))
        mint = min(t)
        for i in range(len(t)):
            t[i] -= mint
        y = []
        for val in file["Flux"].values:
            y.append(val)

        # Frequenzbereich für 3,17 bis 50 Jahre:
        min_freq = 1 / (20 * 365.25 * 24 * 3600)  #  (30 Jahre)
        max_freq = 1 / (2 * 365.25 * 24 * 3600)  #  (2 Jahre)

        frequency, power = LombScargle(t, y).autopower(
            minimum_frequency=min_freq,
            maximum_frequency=max_freq,
            samples_per_peak=40
        )
        max_power_idx = np.argmax(power)
        dominant_freq = frequency[max_power_idx]
        dominant_power = power[max_power_idx]

        # ===== Sin Plot =====

        time_diff = (t[-1] - t[0])
        amp = curve[value].max() - curve[value].min()
        peak = 0
        amp_diff = 0
        x = [0,0]
        y = [0,0]
        try:
            params, params_covariance = optimize.curve_fit(BasicCalcs.fit_func_sin, t, file[value].values,
                                                           p0=[amp, 1 / time_diff, 0, file[value].mean()],
                                                           maxfev=100000)  # a * np.sin(b * x + c) + d
            x = np.linspace(min(t), max(t), 10000)
            y = BasicCalcs.fit_func_sin(x, *params)

            Tsin = abs(params[1])
            if plot:
                return x, y, Tsin
            x = BasicCalcs.Unix_in_Datetime(x)
            T, peaks = FindActive.FourierLombScargle(name)
            for i in range(len(T)):
                if T[i] / Tsin < 1.2 and T[i] / Tsin > 0.8:
                    peak = peaks[i]
                    amp_diff = y.max() - y.min()

        except:
            print("kein sin")
            if plot:
                return [0,0],[0,0],0


        return x,y,dominant_power,amp_diff,dominant_freq,peak # Maximaler Power-Wert im gewählten Bereich


    def periodic3(name): # kopie von periodic als backup
        curve = FileManager.load_data(name)
        file2 = BasicCalcs.rolling_mid(curve)
        file2 = BasicCalcs.normalize_null(curve)
        # file2 = BasicCalcs.rolling_mid(file2)
        file2.dropna(inplace=True)
        maximum = curve[value].max()
        numeric_index = BasicCalcs.Datetime_in_Unix(file2.index)

        # ===== FIT ======
        if len(numeric_index) < 20:
            return [], [], 0, 0, 0, 0
        time_diff = (numeric_index[-1] - numeric_index[0])
        amp = curve[value].max() - curve[value].min()

        try:
            params, params_covariance = optimize.curve_fit(BasicCalcs.fit_func_sin, numeric_index, file2[value].values,
                                                           p0=[amp, 1 / time_diff, 0, file2[value].mean()],
                                                           maxfev=100000)  # a * np.sin(b * x + c) + d
        except:
            return [], [], 0, 0, 0, 0
        x = np.linspace(min(numeric_index), max(numeric_index), 10000)
        y = BasicCalcs.fit_func_sin(x, *params)

        Tsin = abs(params[1])
        if plot:
            return x, y, Tsin

        x = BasicCalcs.Unix_in_Datetime(x)
        T, peaks = FindActive.FourierLombScargle(name)
        # console.print(f"Tsin: {Tsin}, T: {T}, Peaks: {peaks}")
        for i in range(len(T)):
            if T[i] / Tsin < 1.2 and T[i] / Tsin > 0.8:
                # console.print(f"Period T: {T[i]} Tsin: {Tsin} T/Tsin: {T[i]/Tsin}")
                return x, y, abs(params[0]), (y.max() - y.min()), (1 / abs(params[1] * 60)), peaks[i]
            else:
                pass
                # console.print(f"Passt nicht {i} Period T: {T[i]} Tsin: {Tsin} T/Tsin: {T[i]/Tsin}")

        return x, y, abs(params[0]), (y.max() - y.min()), (1 / abs(params[1] * 60)), 0  # T in sekunden
    
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
            datei.write("name,activity,R,activity*R,cuts,amplitude,amp_diff,period,periodicpercent,Dt,std,up,down,mean,peakA,peakC,pointCount,StartEndDiff,redshift,periodicFast,periodicFastFreq,magnitude\n")
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
if config["Plots"]["ShowGroupPlot"] or config["Plots"]["ShowGroupPlotAll"]:
    correlation = pd.DataFrame(columns=["xName","yName","correlation","removed","len","delete"])

    if config["Plots"]["ShowGroupPlotAll"] == False:
        Plots.standart1()
    else:
        parameters = ["F_var", "R", "F_and_R", "cuts", "amplitude","amp_diff", "T", "periodicpercent", "Dt", "std", "up", "down", "mean", "peakA", "peakC", "lange", "StartEndDiff", "redshift","periodicFast","magnitude","periodicFastFreq"]
        for x in range(len(parameters)):
            for y in range(x+1,len(parameters)):
                corr,deleted,lange = Plots.standart1(x=parameters[x], y=parameters[y], plot=False)
                new_row = pd.DataFrame([{"xName": parameters[x], "yName": parameters[y], "correlation": abs(corr), "removed": deleted,"len":lange,"delete":False}])
                correlation = pd.concat([correlation,new_row], ignore_index=True)
                new_row = pd.DataFrame([{"xName": parameters[y], "yName": parameters[x], "correlation": abs(corr), "removed": deleted,"len":lange,"delete":True}])
                correlation = pd.concat([correlation,new_row], ignore_index=True)
            corr, deleted, lange = Plots.standart1(x=parameters[x], y=parameters[x], plot=False)
            new_row = pd.DataFrame([{"xName": parameters[x], "yName": parameters[x], "correlation": abs(corr),"removed": deleted, "len": lange,"delete":True}])
            correlation = pd.concat([correlation, new_row], ignore_index=True)
        correlation.sort_values(by=["correlation"], ascending=False, inplace=True)
        correlation.reset_index(drop=True, inplace=True)

        correlation = correlation[correlation["removed"] < 500]

        heatmap_data = correlation.pivot(index="xName",columns="yName",values="correlation")
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, cmap='Blues', linewidths=.5, linecolor='black', cbar_kws={'label': 'Correlation'})
        plt.title('Heatmap of Values', fontsize=18)
        plt.xlabel('X Name', fontsize=14)
        plt.ylabel('Y Name', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.show()
        plt.close()
        correlation = correlation[correlation["delete"] == False]
        for i in range(40):
            if correlation.iloc[i]["yName"] == "magnitude":
                Plots.standart1(correlation.iloc[i]["xName"],correlation.iloc[i]["yName"],plot=True)

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
    df_sorted.to_csv("sortedcurves.csv")

    if config["Plots"]["statisticalDistribution"]:
        Plots.statisticalDistribution(df_sorted)
    if config["Plots"]["ShowClassifyPlot"]:
        console.print(df_sorted.to_string())
    if config["Plots"]["sortedPlot"]:
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
   
if config["Plots"]["showMatchesOld"]:
    files = pd.read_csv("sortedcurves.csv")
    matches = []
    for val in files["Unnamed: 0"].values:
        _, match,category = FindActive.compareCategories(val)
        matches.append({"name": val, "match": match, "category": category})
    matches_df = pd.DataFrame(matches)
    # Count occurrences of True and False
    Plots.matchesPlot(matches_df)
if config["Plots"]["showMatches"]["show"]:
    df, df_names = FindActive.GetCategroies()
    column_sums = df.sum()
    normalized_df = df / column_sums
    if config["Plots"]["showMatches"]["showPlot"]:
        Plots.plotMateches(normalized_df,df_names)

    SN = df_names["SN"]["inactive"]

    plot = SN

    # for val in plot:
    #     Plots.plot_curves(val)



"""
    TODO: group plot correlation für alle kombinationen durchrechnen lassen und dann ausgeben
    TODO: SDSSJ15242 sollte langzeit periodisch sein
    TODO: langsame periode soll abgeschnitten werden und dann in fast übergehen
    
    TODO: fehlen von kategorien anzeigen yes
    TODO: abs magnitude runterladen
    
    TODO: Lichtkurven mit webscrapping nach echter klassifizierung absuchen
"""


