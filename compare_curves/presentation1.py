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

path_final = "/final_light_curves"
path_old = "/light_curves"
value = "Flux"

names = pd.DataFrame(columns=["name","id"])
names = pd.concat([names, pd.DataFrame({"name":["NGC 1566", "NGC 4593", "NGC 4151"],"id":[661431822098,661431908458,42949788551]})], ignore_index=True)

def neumann_cam_shift(data,curve):

    #find overlapp 
    start_overlap = max(data["JD"].min(), curve["JD"].min())
    end_overlap = min(data["JD"].max(), curve["JD"].max())
    if data["JD"].max() < curve["JD"].min(): # ! Verschiebt Kurve mit mean falls es keine Überlagerung gibt
        len_data = len(data["JD"])
        len_curve = len(curve["JD"])
        points = 40 
        if min(len_data,len_curve) < 40:
            points = min(len_data,len_curve)
        mean_existing = data[value][-points:].mean()
        mean_shift_curve = curve[value][:points].mean()
        shift_mean = mean_existing - mean_shift_curve
        curve[value] = curve[value] + shift_mean
        return curve
    if len(curve["JD"]) < 2:
        len_data = len(data["JD"])
        points = 40 
        if len_data < 40:
            points = len_data
        mean_existing = data[value][-points:].mean()
        mean_shift_curve = curve[value].mean()
        shift_mean = mean_existing - mean_shift_curve
        curve[value] = curve[value] + shift_mean
        return curve
            
    if end_overlap < start_overlap:
        return curve
    
    main_curve = data[(data['JD'] >= start_overlap) & (data['JD'] <= end_overlap)].copy()
    fit_curve = curve[(curve['JD'] >= start_overlap) & (curve['JD'] <= end_overlap)].copy()
    main_curve.reset_index(drop=True, inplace=True)
    fit_curve.reset_index(drop=True, inplace=True)
    
    if len(main_curve["JD"]) < 1: # falls die Kurve genau in eine Lücke ohne Daten fällt
        len_overlap = len(data[(data['JD'] <= start_overlap)])
        if len_overlap < 10:
            len_overlap = len(data[(data['JD'] <= start_overlap)])
        else :
            len_overlap = 10
        mean_not_overlapping_curve = data.loc[(data['JD'] <= start_overlap), value][-len_overlap:].mean()
        mean_shift_curve = curve[value].mean()
        shift_mean = mean_not_overlapping_curve - mean_shift_curve
        curve[value] = curve[value] + shift_mean
        return curve
    #NGC4395 als beispiel für eine Lücke
    
    fit_curve_backup = fit_curve.copy()
    mean_plot_curve = fit_curve.copy() #! löschen
    #R = np.arange(-20,20,0.01) # ! Funktion für die Range schreiben um minimum schneller zu finden
    T = []
    go = True
    shift = 0
    R = []
    step = 1
    T_save = 10e10
    change = True # True == <= und False == >
    while True:
        T.append(0)
        R.append(0)
        fit_curve = fit_curve_backup.copy()
        fit_curve[value] = fit_curve[value] + shift
        calculate_curve = pd.concat([fit_curve,main_curve])
        calculate_curve.sort_values(by='JD', inplace=True)
        calculate_curve.reset_index(drop=True, inplace=True)
        
        for i in range(len(calculate_curve)-1):
            T[-1] += (calculate_curve[value][i] - calculate_curve[value][i+1])**2 
        if abs(step) < 0.01 or (T[-1] == 0 and abs(shift) >= 100):
            R[-1] = shift
            break
        
        if T[-1] > T_save:
            step = -step/2
        T_save = T[-1]
        R[-1] = shift
        shift = shift + step
            
            
    shift = R[T.index(min(T))]
    # TODO ZUM TESTEN MEAN BERECHNET -> SOLLTE NOCH GELÖSCHT WERDEN
    mean1 = main_curve[value].mean()
    mean2 = mean_plot_curve[value].mean()
    mean_diff = mean1 - mean2
    mean_plot_curve[value] = mean_plot_curve[value] + mean_diff
    # ============================================================
    if False:
        for shift in R:
            T.append(0)
            fit_curve = fit_curve_backup.copy()
            fit_curve[value] = fit_curve[value] + shift
            calculate_curve = pd.concat([fit_curve,main_curve])
            calculate_curve.sort_values(by='JD', inplace=True)
            calculate_curve.reset_index(drop=True, inplace=True)
            
            for i in range(len(calculate_curve)-1):
                T[-1] += (calculate_curve[value][i] - calculate_curve[value][i+1])**2
            
        shift = R[T.index(min(T))]
    
    all_data = pd.concat([data,curve])
    all_data.sort_values(by='JD', inplace=True)
    fit_curve[value] = fit_curve_backup[value] + shift
    curve[value] = curve[value] + shift
    return curve

def shift_cam_and_filters(file):
    cameras = file["Camera"].unique() 
    min = []
    for c in cameras:
        min.append(file.loc[file["Camera"] == c, "JD"].min())

    #Kameras sortieren
    for i in range(len(min)):   
        for j in range(len(min)-i-1):
            if min[j] > min[j+1]:
                min[j], min[j+1] = min[j+1], min[j]
                cameras[j], cameras[j+1] = cameras[j+1], cameras[j]
    
    main_curve = file[file["Camera"] == cameras[0]].copy()
    for i in range(1,len(cameras)):
        #print(f"cam: {cameras[i]} nr: {i}")
        fit_curve = file[file["Camera"] == cameras[i]].copy()
        
        # Check ob es größere Lücken oder Sprünge gibt -> Dann Lichtkurve weiter unterteilen # ! Wenn ein Cluster stark verschoben ist -> wird als seperate Kurve behandelt
            

        fit_curve = neumann_cam_shift(main_curve,fit_curve)
        main_curve = pd.concat([main_curve, fit_curve], ignore_index=True)
        main_curve.sort_values(by='JD', inplace=True)
    return main_curve

def read_data_from_jd(filepath):
    
    if filepath == ".DS_Store" or filepath == "presentation1.py" or filepath == "name_id.csv" or filepath == "active_galaxies.csv":
        return None, False
    
    start_line = 0
    JD = False
    with open(filepath, 'r') as file:        
        for i, line in enumerate(file):
            if i > 2 and line.startswith("JD"):
                start_line = i
                JD = True
                break
            elif i < 2 and line.startswith(","):
                start_line = 0
                JD = False
                break
            elif i < 2 and line.startswith("JD"):
                start_line = 0
                JD = True
                break
    
    # Lies die Datei ab der gefundenen Zeile ein

    df = pd.read_csv(filepath, skiprows=range(start_line))
    return df, JD
def rolling_mid(file, rolling_time ="28D"):
    # Berechne den gleitenden Mittelwert über 28 Tage
    file2 = file.copy() 
    file2.set_index('JD', inplace=True)
    file2.sort_index(inplace=True)
    file2[value] = file2[value].rolling(window=rolling_time, center = False, min_periods = 4).mean()
    #file2.dropna(inplace=True) #! Sinvoll?
    file2.reset_index(inplace=True) 
    return file2

files = listdir()
curves = []
curves_name = []
for f in tqdm(files):
    file, JD = read_data_from_jd(f)
    if type(file) == type(None):
        continue
    if JD:
        file["JD"] = pd.to_datetime(file['JD'], origin='julian', unit='D')
    else:
        file["JD"] =  pd.to_datetime(file['JD'])
    file = shift_cam_and_filters(file)
    curves.append(file)
    curves_name.append(f)



for id in names["id"]:
    index = curves_name.index(f"{id}-pre_cleaned.csv")
    index2 = curves_name.index(f"{id}-shift.csv")
    index3 = curves_name.index(f"{id}-no_shift.csv")
    index4 = curves_name.index(f"{id}-light-curves.csv")
    c = curves[index]
    x1 = curves[index2]
    x2 = curves[index3]
    x3 = curves[index4]
    x1 = neumann_cam_shift(c,x1)
    x2 = neumann_cam_shift(c,x2)
    x3 = neumann_cam_shift(c,x3)
    curves[index2] = x1
    curves[index3] = x2
    curves[index4] = x3

    


title = ""
color = "yellow"
zorder = 0
for id in names["id"]:
    for name, curve in zip(curves_name, curves):
        if id != int(name.split("-")[0]):
            continue
        title = names[names["id"] == int(name.split("-")[0])]["name"].values[0]
        print(name)
        if "pre_cleaned" in name:
            print(f"1 name: {name}")
            color = "red"
            label = "Image Subtraction"
            zorder = 3
        elif "no_shift" in name:
            print(f"2 name: {name}")
            color = "blue"
            label = "ohne cuts"
            zorder = 2
        elif "shift" in name:
            print(f"3 name: {name}")
            color = "green"
            label = "mit cuts geshiftet"
            zorder = 1
        elif "light-curves" in name:
            print(f"4 name: {name}")
            color = "orange"
            label = "original (V/g shift)"
            zorder = 0
            #continue
        else:
            label = "None"
        file2 = rolling_mid(curve, "30D")
        plt.scatter(curve["JD"],curve["Flux"], s = 12,label = label, c=color,marker = "x", zorder=zorder,alpha=0.2)
        plt.plot(file2["JD"],file2["Flux"],label = label, c=color, zorder=zorder,alpha=1)
        
    plt.title(title)
    plt.xlabel("JD")
    plt.ylabel("Flux")
    plt.grid()
    #plt.gca().set_facecolor('gray')
    plt.legend()
    plt.show()
    plt.close()
        
