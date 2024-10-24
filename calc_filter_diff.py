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
# ====== CHANGE =====
path = "light_curves/"
#path = "compare_curves/"
value = "Flux"
mid = 12
treshold_F_var = 0.0025 # ab wann eine Galaxie aktiv ist
treshold_R = 1.2 # ab wann eine Galaxie aktiv ist
plot_von_neumann_shift = False
plot_curves_general = False


galaxy_cuts = [661430607288,635656070242,661430304024,661432030252,661431933652,661431844884,377957862504,214749149546,532576412113,128850141936,120260119268,661430499735,283467877327,661430490724,661430417501,446677648638]


# ====== DO NOT CHANGE =====
filename_path = "light_curves/name_id.csv"
filename = pd.read_csv(filename_path, usecols=lambda column: column != 'Unnamed: 0')
filename.reset_index(drop=True, inplace=True)
current_ID = 0
current_cuts = 0

if not os.path.exists(path+"active_galaxies.csv"):
    # Datei erstellen
    with open(path+"active_galaxies.csv", 'w') as datei:
        datei.write("ID,name,acivity,R,times,cuts\n")  # Leere Datei erstellen oder optionalen Text hineinschreiben
    print("Datei wurde erstellt.")
with open(path+"active_galaxies.csv", 'w') as datei:
    datei.write("ID,name,acivity,R,times,cuts\n")
    
plot_cuts = pd.DataFrame(columns = ["start","cam","shift"],dtype = "float")
plot_cuts = plot_cuts.astype({'start': 'datetime64[ns]', "cam": "str"})
#=======================================================


def funcion_time(func): # für eine bessere übersicht beim ausführen der Aufgaben
    def warpper(*args,**kwargs):
        print(f"\n\n========== Aufgabe {func.__name__} ==========\n\n")
        start_time = time.time()
        result = func(*args,**kwargs)
        end_time = time.time()
        print(f"\n\n========== Ende Aufgabe {end_time-start_time} s ==========\n\n")
        return result
    return warpper


def read_data_from_jd(filepath):
    with open(filepath, 'r') as file:
        # Suche die Zeile, die mit "JD" beginnt
        for i, line in enumerate(file):
            if line.startswith("JD"):
                start_line = i
                break
    
    # Lies die Datei ab der gefundenen Zeile ein
    df = pd.read_csv(filepath, skiprows=range(start_line))
    return df

def neumann_remove(df, threshold=3):
    
    def prepare_data_time():
        mean = df[value].rolling(window=mid, center=True).median() # median besser als mean?
        mean_mean = mean.mean()
        std = df[value].rolling(window=mid, center=True).std()
        std_mean = std.mean()
        mean = np.where(np.isnan(mean), mean_mean, mean)
        std = np.where(np.isnan(std), std_mean, std)
        df2 = df.copy()
        for i in reversed(range(0,len(df))):
            if df2.at[i,value] > ((mean[i] + std[i])*1.01) or df2.at[i,value] < ((mean[i] - std[i])*0.99):
                df2.drop(i, inplace=True)
        df2.reset_index(drop=True, inplace=True)
        # neumann berechnung
        T.append(0)
        T_test.append(0)
        for i in range(len(df2)-1):
            T[-1] += (df[value][i] - df2[value][i+1])**2
        T[-1] = T[-1] * df2[value].std()
        return df2
    def prepare_data_difference():
        mean = df[value].rolling(window=20, center=False).median() # median besser als mean?
        mean_mean = mean.mean()
        std = df[value].rolling(window=20, center=False).std()
        std=std**2
        std_mean = std.mean()
        mean = np.where(np.isnan(mean), mean_mean, mean)
        std = np.where(np.isnan(std), std_mean, std)
        df2 = df.copy()
        upper = 1+(mid - 4)/10000
        lower = 1-(mid - 4)/10000
        drop = False
        for i in reversed(range(0,len(df))):
            if df2.at[i,value] > ((mean[i] + std[i]*upper)) or df2.at[i,value] < ((mean[i] - std[i]*lower)):
                df2.drop(i, inplace=True)
                drop = True
        df2.reset_index(drop=True, inplace=True)
        # neumann berechnung
        if drop == False and len(T) >= 1:
            T.append(T[-1])
            T_test.append(T_test[-1])
        else:
            T.append(0)
            T_test.append(0)
            for i in range(len(df2)-1):
                T[-1] += (df[value][i] - df2[value][i+1])**2
            T[-1] = T[-1] * df2[value].std()**2
        return df2
    # Mittelwert und Standardabweichung berechnen und neue Kurve erstellen
    mean = df[value].mean()
    std = df[value].std()
    df[value] = (df[value]-mean)/std
    df2 = df.copy()
    #für zweite kurve daten punkte löschen
    r = range(5,500)
    T=[]
    T_test=[]
    for mid in tqdm(r):
        #prepare_data_time()
        prepare_data_difference()
    mid = r[T.index(min(T))]
        
    df2=prepare_data_difference()
    #df2=prepare_data_time()
    if True: #plot zeigen
        print(f"\noptimale Werte bei upper: {1+(mid - 4)/10000} lower: {1-(mid - 4)/10000}\n")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        T.pop()
        T_test.pop()
        df2.sort_values(by='JD', inplace=True)
        df.sort_values(by='JD', inplace=True)
        # Plot 1: r vs T
        ax1.plot(r, T)
        if 0 not in T_test:
            ax1.plot(r, T_test, label='test', color = "black", alpha=0.5)
        ax1.set_title('Plot 1: r vs T')
        ax1.set_xlabel('r')
        ax1.set_ylabel('T')
        ax1.vlines(mid, color='red', label='min T', ymin = min(T), ymax = max(T))
        ax1.grid(True)
        ax2.grid(True)
        ax2.plot(df['JD'], df[value], label='a',alpha = 0.5)
        ax2.plot(df2['JD'], df2[value], label='new',alpha = 0.5)
        ax2.set_title(f"\noptimale Werte bei upper: {1+(mid - 4)/10000} lower: {1-(mid - 4)/10000}\n")
        ax2.set_xlabel('Time')
        ax2.set_ylabel('wert')
        ax2.legend()  # Legende hinzufügen

        # Layout anpassen und anzeigen
        plt.tight_layout()
        plt.show()
        plt.close()
    df2[value] = (df2[value]*std + mean)
    return df2

def remove_outliers_mad(file, threshold=3):
    def neumann_diff(a,b, threshold=3): # format von a: pandas [times, values]
        #* ==== HOW TO USE ====
        """
        a = file.loc[(file[filter] != x), ['JD', value]].copy()
        a.rename(columns={value: 'value'}, inplace=True)
        b = file.loc[(file[filter] == x), ['JD', value]].copy()
        b.rename(columns={value: 'value'}, inplace=True)
        file.loc[(file[filter] == y), 'JD'] = file.loc[(file[filter] == y), 'JD'] + pd.Timedelta(days=neumann_diff(a,b))
        file.sort_index(inplace=True) 
        """
        # TODO normieren der Lichtkurven -> mean = 0, std = 1
        mean_a = a["wert"].mean()
        std_a = a["wert"].std()

        mean_b = b["wert"].mean()
        std_b = b["wert"].std()

        t_min = abs(a["JD"].min() - b["JD"].max()).days
        t_max = abs(a["JD"].max() - b["JD"].min()).days

        if t_min > 500: # sonst dauert es zu lange
            t_min = 500
        if t_max > 500:
            t_max = 500

        a["wert"] = (a["wert"]-mean_a)/std_a
        b["wert"] = (b["wert"]-mean_b)/std_b
        # Daten einsortieren
        F = pd.concat([a, b]).copy()
        #F = F.sort_values(by='JD').reset_index(drop=True)
        F = F.sort_values(by='JD').reset_index(drop=True)
        # um tau verschobene Daten
        T = []
        T_test = []
        r = range(-t_min,t_max)
        for tau in tqdm(r):
            c = b.copy()
            c["JD"] = c["JD"] + pd.Timedelta(days=tau)
            F_tau = pd.concat([a, c]).copy()

            
            #F_tau = F_tau.sort_index().reset_index(drop=True)
            F_tau = F_tau.sort_values(by='JD').reset_index(drop=True)
            #T(tau) berechnen
            T.append(0)
            T_test.append(0)
            for i in range(len(F)-1):
                T[-1] += (F["wert"][i] - F_tau["wert"][i+1])**2
            T[-1] = T[-1] * 1/(len(F)-1) 
            T_test[-1] = T[-1]
            T[-1] = T[-1] + abs(tau*0.0005)**2
        time_shift = r[T.index(min(T))]
        if time_shift == - t_min:
            time_shift = 0

        #plotten


        if True: #plot zeigen
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
            c = b.copy()
            c["JD"] = c["JD"] + pd.Timedelta(days=time_shift)
            c.sort_values(by='JD', inplace=True)
            a.sort_values(by='JD', inplace=True)
            # Plot 1: r vs T
            
            ax1.plot(r, T)
            ax1.plot(r, T_test, label='test', color = "black", alpha=0.5)
            ax1.set_title(f"Galaxy: {get_galaxy_name()}")
            ax1.set_xlabel('r')
            ax1.set_ylabel('T')
            ax1.vlines(time_shift, color='red', label='min T', ymin = min(T), ymax = max(T))
            ax1.grid(True)
            ax2.grid(True)
            ax2.plot(a['JD'], a['wert'], label='a')
            ax2.plot(c['JD'], c['wert'], label='new')
            ax2.plot(b['JD'], b['wert'], label='original', color='black', alpha=0.5)
            ax2.set_title('Plot 2: verschoben um {} Tage'.format(time_shift))
            ax2.set_xlabel('Time')
            ax2.set_ylabel('wert')
            ax2.legend()  # Legende hinzufügen

            # Layout anpassen und anzeigen
            plt.tight_layout()
            plt.show()
            plt.close()
        return time_shift # Zeitdifferenz

    def move(filter,elements):
        stop = False
        for index, x in enumerate(elements[:-1]):
            for y in elements[index+1:]:
                if x==y:
                    stop = True
                    break

                # Extrahiere JD-Werte für beide Filter
                jd_v = file.loc[(file[filter] == x), 'JD'].copy()
                jd_g = file.loc[(file[filter] == y), 'JD'].copy()

                # Berechne den Überlappungsbereich
                start_overlap = max(jd_v.min(), jd_g.min())
                end_overlap = min(jd_v.max(), jd_g.max())
                if end_overlap < start_overlap:
                    break
                # time shift
                if False:
                    a = file.loc[(file[filter] != x), ['JD', value]].copy()
                    a.rename(columns={value: 'wert'}, inplace=True)
                    b = file.loc[(file[filter] == x), ['JD', value]].copy()
                    b.rename(columns={value: 'wert'}, inplace=True)
                    time_add = pd.Timedelta(days=neumann_diff(a,b))
                    file.loc[(file[filter] == x), 'JD'] = file.loc[(file[filter] == x), 'JD'] + time_add
                    file.sort_index(inplace=True)

                    # erneutes bestimmen des Überlappungsbereichs da file sich geändert hat
                    jd_v = file.loc[(file[filter] == x), 'JD'].copy()
                    jd_g = file.loc[(file[filter] == y), 'JD'].copy()
                    start_overlap = max(jd_v.min(), jd_g.min())
                    end_overlap = min(jd_v.max(), jd_g.max())
                    if end_overlap < start_overlap:
                        break

                overlap_indices = file[(file['JD'] >= start_overlap) & (file['JD'] <= end_overlap)].index.copy()

                mean_g = file.loc[(file[filter] == y) & (file.index.isin(overlap_indices)), value].mean()
                mean_v = file.loc[(file[filter] == x) & (file.index.isin(overlap_indices)), value].mean()
                mean_diff = mean_v-mean_g
                file.loc[(file[filter] == y), value] += mean_diff
            if stop:
                break
        return file 
    position = len(file)
    if True:
        cams = file["Camera"].unique().copy()
        for _ in range(1):
            for c in cams:
            #for c in ["V","g"]:
                df = file[file.Camera == c].reset_index(drop=True)
                df = df.sort_values(by='JD').reset_index(drop=True)
                
                #time shift
                if False:
                    if c == "g":
                        a = file.loc[(file["Filter"] != c), ['JD', value]].copy()
                        a.rename(columns={value: 'wert'}, inplace=True)
                        b = file.loc[(file["Filter"] == c), ['JD', value]].copy()
                        b.rename(columns={value: 'wert'}, inplace=True)
                        df["JD"] = df["JD"] + pd.Timedelta(days=neumann_diff(a,b))
                        df.sort_index(inplace=True)

            
                if True:    
                    mean = df[value].rolling(window=mid, center=True).mean() # median besser als mean?
                    #print(mean.to_string())
                    mean_mean = mean.mean()
                    std = df[value].rolling(window=mid, center=True).std()
                    std_mean = std.mean()
                    full_std = df[value].std()
                    mean = np.where(np.isnan(mean), mean_mean, mean)
                    std = np.where(np.isnan(std), std_mean, std)

                    for i in reversed(range(0,len(df))):
                        #if df[value][i] > mean[i] + mean[i]*0.02 or df[value][i] < mean[i] - mean[i]*0.02:
                        if df.at[i,value] > ((mean[i] + std[i])*1.01) or df.at[i,value] < ((mean[i] - std[i])*0.99):
                            df.drop(i, inplace=True)
                df.reset_index(drop=True, inplace=True)  # Indizes nach dem Löschen zurücksetzen
                file = pd.concat([file, df], ignore_index=True)
                
            file.drop(index=range(0, position), inplace=True)
            file.reset_index(drop=True, inplace=True)
    # ==== g Filter verschieben ==== 

    #move("Filter",["V","g"])
    cameras = file["Camera"].unique()    
    #move("Camera",cameras)

    return file
def get_galaxy_name():
    if current_ID not in filename["ID"].values:
        return "ID not found - ID: {}".format(current_ID)
    name = filename.loc[filename["ID"] == current_ID, "name"]
    return name.values[0]
    
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
    if plot_von_neumann_shift: #plot zeigen
        get_galaxy_name()
        print(f"\noptimaler shift bei {shift} (mean shift: {mean_diff} -> Diff: {shift-mean_diff} \nmit T min = {min(T)} bei index: {T.index(min(T))} bei R: {R[T.index(min(T))]}\n")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        # Plot 1: r vs T
        ax1.scatter(R, T)
        ax1.set_title(f"Galaxy: {get_galaxy_name()}")
        ax1.set_xlabel('r')
        ax1.set_ylabel('T')
        ax1.vlines(shift, color='red', label='min T', ymin = min(T), ymax = max(T))
        ax1.grid(True)
        ax2.grid(True)
        ax2.plot(data['JD'], data[value], label='existing',alpha = 0.8, color = "green")
        ax2.plot(curve['JD'], curve[value], label='new',alpha = 0.8, color = "red")
        ax2.plot(mean_plot_curve['JD'], mean_plot_curve[value], label='mean shift',alpha = 0.8,color = "orange")
        ax2.plot(all_data['JD'], all_data[value], label='full plot',alpha = 0.3,color = "blue")
        ax2.set_title(f"\noptimaler shift bei {shift}\n")
        ax2.set_xlabel('Time')
        ax2.set_ylabel('wert')
        ax2.legend()  # Legende hinzufügen

        # Layout anpassen und anzeigen
        plt.tight_layout()
        plt.show()
        plt.close()
    return curve
def shift_cam_and_filters(file):
    cameras = file["Camera"].unique() 
    filters = file["Filter"].unique()
    if len(cameras) <= 1:
        return file
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
        if True:
            fit_curve.reset_index(drop=True, inplace=True)
            fit_curve.sort_values(by='JD', inplace=True)
            global plot_cuts
            curve_splitter = pd.DataFrame(columns=["cut", "mean", "std"], dtype='float')   
            curve_splitter = curve_splitter.astype({'cut': 'int64'})         
            start = 0
            # ==== Bedingung 1 
            for i in range(len(fit_curve["JD"])-1):
                if (fit_curve["JD"][i+1] - fit_curve["JD"][i] > pd.Timedelta(days=30)) or (abs(fit_curve[value][i+1] - fit_curve[value][i]) > fit_curve[value].std()): # Maximale Lückengröße
                    if len(fit_curve[value][start:i]) <2: #! entfernen? (wenn in einem einzelnen Zeitraum weniger als 2 Werte vorhanden sind)
                        start = i+1
                        continue
                    curve_splitter = pd.concat([curve_splitter, pd.DataFrame([{"cut":i,"mean":fit_curve[value][start:i].mean(),"std":fit_curve[value][start:i].std()}])], ignore_index=True)                
                    start = i+1
            # ==== Bedingung 2
            # avg_days = 4
            # for i in range(avg_days,len(fit_curve["JD"])-avg_days - 1):  
            #     mean = fit_curve[value][i-avg_days:i].mean()
            #     if abs(mean - fit_curve[value][i+1:i+1+avg_days].mean()) > (fit_curve[value][i-avg_days:i].std() - fit_curve[value][i+1:i+1+avg_days].std()).mean():
            #         for k in range(i,len(fit_curve["JD"])):
                        


                
            # ================
                                        
            mean_std = curve_splitter["std"].mean()
            new_curves = pd.DataFrame(columns = ["cut_start"])
            
            start_data = {"cut_start":[0,len(fit_curve[value])]}
            new_curves = pd.DataFrame(start_data)

            for i in range(1,len(curve_splitter["cut"])):
                # ==== Bedingung 1 
                if (curve_splitter["std"][i-1] < mean_std*2) and (curve_splitter["std"][i] < mean_std*1.5) and ((curve_splitter["mean"][i] > curve_splitter["mean"][i-1]*1.1) or (curve_splitter["mean"][i] < curve_splitter["mean"][i-1]*0.9)): 
                    new_curves = pd.concat([new_curves, pd.DataFrame([{"cut_start":curve_splitter["cut"][i-1]}])], ignore_index=True)
                    if len(plot_cuts["cam"]) == 0:
                        plot_cuts = pd.DataFrame([{"start":fit_curve.loc[curve_splitter["cut"][i-1]+1,"JD"],"cam":fit_curve.loc[curve_splitter["cut"][i],"Camera"],"shift":curve_splitter["mean"][i] - curve_splitter["mean"][i-1]}])
                    else:
                        plot_cuts = pd.concat([plot_cuts, pd.DataFrame([{"start":fit_curve.loc[curve_splitter["cut"][i-1]+1,"JD"],"cam":fit_curve.loc[curve_splitter["cut"][i],"Camera"],"shift":curve_splitter["mean"][i] - curve_splitter["mean"][i-1]}])], ignore_index=True)
                # ==== Bedingung 2
                
                # =================
            new_curves.sort_values(by='cut_start', inplace=True)
            new_curves.reset_index(drop=True, inplace=True)
            for i in range(1,len(new_curves["cut_start"])):
                fit_curve2 = neumann_cam_shift(main_curve,fit_curve.iloc[new_curves["cut_start"][i-1]:new_curves["cut_start"][i]].copy())
                main_curve = pd.concat([main_curve, fit_curve2], ignore_index=True)
                main_curve.sort_values(by='JD', inplace=True)
            global current_cuts
            current_cuts = len(new_curves["cut_start"]) - 2
            
        else:  
            fit_curve = neumann_cam_shift(main_curve,fit_curve)
            main_curve = pd.concat([main_curve, fit_curve], ignore_index=True)
            main_curve.sort_values(by='JD', inplace=True)
    return main_curve
    
# Beispiel: Entfernen von Ausreißern aus einer Spalte 'Flux'
def rolling_mid(file, rolling_time ="28D"):
    # Berechne den gleitenden Mittelwert über 28 Tage
    file2 = file.copy() 
    file2.set_index('JD', inplace=True)
    file2.sort_index(inplace=True)
    file2[value] = file2[value].rolling(window=rolling_time, center = False, min_periods = 4).mean()
    #file2.dropna(inplace=True) #! Sinvoll?
    file2.reset_index(inplace=True) 
    return file2

class find_active:
    def peak_to_peak_amplitudes(file): # returns R
        return file[value].max() / file[value].min()
    def mean(file):
        return file[value].mean()
    def std(file):
        return file[value].std()
    def delta(file):
        sum = 0
        length = len(file[value])
        for i in range (length):
            sum += file[value + " Error"][i]**2 #! Auch normieren
        try:
            sum = np.sqrt(sum/length)
        except:
            sum = 0
        return sum # nicht quadriert
    def fractional_variation(file):
        file2 = file.copy() #! Lichtkurve muss normalisiert sein damit Kurven mit hohem Flux nicht höhere Bewertung bekommen?
        shift = 0
        if file2[value].min() <= 0:
            shift = file2[value].min() 
            file2[value] = file2[value] - shift + 1
        file2[value] = file2[value]/file2[value].max()
        file2[f"{value} Error"] = file2[f"{value} Error"]/file2[value].max()
        R = find_active.peak_to_peak_amplitudes(file2)
        acivity = (find_active.std(file2)**2-find_active.delta(file2)**2) / find_active.mean(file2)
        find_active.group_galaxies(acivity,R)
    def group_galaxies(acivity,R):
        file_path = path+"active_galaxies.csv"

        with open(file_path, 'r') as datei:
            data = datei.read()
        if str(current_ID) in data:
            return
        with open(file_path, 'a') as datei:
            global current_cuts
            datei.write(f"{current_ID},{get_galaxy_name()},{acivity},{R},{acivity*R},{current_cuts}\n")
        return

def save_final_curves(file):
    file.to_csv(f"final_light_curves/{get_galaxy_name()}.csv",index=False)
    return
def visualize1(file):
    x_org, y_org, cam = file["JD"].copy(), file[value].copy(), file["Camera"].copy()
    file = remove_outliers_mad(file)
    file = shift_cam_and_filters(file)
    save_final_curves(file)
    #file = remove_outliers_mad(file)
    find_active.fractional_variation(file)
    #file = neumann_remove(file)
    x1,y1 = file["JD"].copy(), file[value].copy()
    cam2, filter2 = file["Camera"].copy(), file["Filter"].copy()
    #file2 = rolling_mid(file,"10D")
    #x2,y2 = file2["JD"].copy(), file2[value].copy() # "JD" ist jetzt der Index!!!
    file2 = rolling_mid(file,"30D")
    x3,y3 = file2["JD"].copy(), file2[value].copy() # "JD" ist jetzt der Index!!!
    #file2 = rolling_mid(file,"90D")
    #x4,y4 = file2["JD"].copy(), file2[value].copy() # "JD" ist jetzt der Index!!!

    # === Farben
    # Filter Farben
    c = []
    for i in file["Filter"]:
        if i == "V":
            c.append("green")
        else:
            c.append("red")

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
    for index, i in enumerate(cam2):
        if file["Filter"][index] == "V":
            c3.append(farben[np.where(cameras == i)[0][0]])
        else:
            c4.append(farben[np.where(cameras == i)[0][0]])
    x_1 = file.loc[file["Filter"] == "V", "JD"].copy()
    x_2 = file.loc[file["Filter"] == "g", "JD"].copy()
    y_1 = file.loc[file["Filter"] == "V", value].copy()
    y_2 = file.loc[file["Filter"] == "g", value].copy()
    # === Plot
    if plot_curves_general:
        galaxy_active = pd.read_csv(path+"/active_galaxies.csv")
        if current_ID in galaxy_active["ID"].values:
            if galaxy_active.loc[galaxy_active["ID"] == current_ID, "acivity"].values[0] <= treshold_F_var and galaxy_active.loc[galaxy_active["ID"] == current_ID, "R"].values[0] <= treshold_R:
                plt.title(f"NICHT VARIABEL Galaxy: {get_galaxy_name()} TH F: {treshold_F_var} Activity F: {round(galaxy_active.loc[galaxy_active['ID'] == current_ID, 'acivity'].values[0]*10000)/10000}\nTR R: {treshold_R} Activity R: {round(galaxy_active.loc[galaxy_active['ID'] == current_ID, 'R'].values[0]*100)/100}")
            else:
                if galaxy_active.loc[galaxy_active['ID'] == current_ID, 'R'].values[0] == np.inf:
                    plt.title(f"VARIABEL Galaxy: {get_galaxy_name()} TH F: {treshold_F_var} Activity F: {round(galaxy_active.loc[galaxy_active['ID'] == current_ID, 'acivity'].values[0]*10000)/10000}\nTR R: {treshold_R} Activity R: inf")
                else:
                    plt.title(f"VARIABEL \nGalaxy: {get_galaxy_name()} TH F: {treshold_F_var} Activity F: {round(galaxy_active.loc[galaxy_active['ID'] == current_ID, 'acivity'].values[0]*10000)/10000}\nTR R: {treshold_R} Activity R: {round(galaxy_active.loc[galaxy_active['ID'] == current_ID, 'R'].values[0]*100)/100}")
        #plt.plot(x2,y2,zorder=4, label="10 Tage")
        #plt.plot(x3,y3,zorder=6, label="30 Tage",color = "red")
        #plt.plot(x4,y4,zorder=5, label="90 Tage")   
        #plt.scatter(x2,y2,marker = "x",zorder=4)  
        #plt.scatter(x1,y1,c = c,zorder=3, alpha=0.3)
        #plt.scatter(x_org,y_org,c = "black", alpha=0.4, zorder=1,marker = "x")
        #plt.scatter(x_org,y_org,c = c2, alpha=0.4, zorder=5, marker = "o") # plot der Orginalpunkte
        plt.scatter(x_1,y_1,c = c3, alpha=0.2, zorder=5, marker = "x") # plot verschobene orginalpunkte
        plt.scatter(x_2,y_2,c = c4, alpha=0.2, zorder=5, marker = "o") # plot verschobene orginalpunkte
        for i in range(len(plot_cuts["cam"])):
            plot_cuts.loc[i, "cam"] = farben[np.where(cameras == plot_cuts["cam"][i])[0][0]]
        for x,color in zip(plot_cuts["start"],plot_cuts["cam"]):
            pass
            plt.axvline(x=x, color=color, linestyle='solid',alpha = 0.5)
        for x,color,shift in zip(plot_cuts["start"],plot_cuts["cam"],plot_cuts["shift"]):
            if len(y_1) < 2 or len(y_2) < 2:
                continue
            plt.text(x, max(max(y_1),max(y_2)), f"{round(shift*100)/100}",color = color)
        #plt.scatter(x1,y1,c = c3,zorder=5, alpha=0.4, marker = marker)
        plt.grid()
        #plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()



def visualize_with_checkbuttons(file):
    x_org, y_org, cam = file["JD"].copy(), file[value].copy(), file["Camera"].copy()
    file = remove_outliers_mad(file)
    file = shift_cam_and_filters(file)
    find_active.fractional_variation(file)
    file2 = rolling_mid(file, "30D")
    x3, y3 = file2["JD"].copy(), file2[value].copy()
    
    # Filter colors
    c = []
    for i in file["Filter"]:
        if i == "V":
            c.append("green")
        else:
            c.append("red")

    # Camera colors
    farben = [
    "blue", "black", "cyan", "magenta", "yellow", "black", "white", "orange", 
    "purple", "brown", "pink", "gray", "olive", "darkblue", "lime", "indigo", 
    "gold", "darkgreen", "teal", "black", "cyan", "magenta", "yellow", "black", "white", "orange", 
    "purple", "brown", "pink", "gray", "olive", "darkblue", "lime", "indigo", 
    "gold", "darkgreen", "teal"
    ]
    cameras = cam.unique()
    c2 = [farben[np.where(cameras == i)[0][0]] for i in cam]
    
    cam2, filter2 = file["Camera"].copy(), file["Filter"].copy()
    c3, c4 = [], []
    for index, i in enumerate(cam2):
        if file["Filter"][index] == "V":
            c3.append(farben[np.where(cameras == i)[0][0]])
        else:
            c4.append(farben[np.where(cameras == i)[0][0]])
    
    x_1 = file.loc[file["Filter"] == "V", "JD"].copy()
    x_2 = file.loc[file["Filter"] == "g", "JD"].copy()
    y_1 = file.loc[file["Filter"] == "V", value].copy()
    y_2 = file.loc[file["Filter"] == "g", value].copy()
    
    # Create plot
    fig, ax = plt.subplots()
    
    scatter1 = ax.scatter(x_1, y_1, c=c3, alpha=0.2, zorder=5, marker="x", label="V Filter")
    scatter2 = ax.scatter(x_2, y_2, c=c4, alpha=0.2, zorder=5, marker="o", label="g Filter")
    
    plt.grid()
    plt.tight_layout()

    # Check button logic
    rax = plt.axes([0.05, 0.4, 0.15, 0.15])  # Position of the CheckButtons widget
    labels = ['V Filter', 'g Filter']
    visibility = [scatter1.get_visible(), scatter2.get_visible()]
    
    check = CheckButtons(rax, labels, visibility)

    def func(label):
        if label == 'V Filter':
            scatter1.set_visible(not scatter1.get_visible())
        elif label == 'g Filter':
            scatter2.set_visible(not scatter2.get_visible())
        plt.draw()

    check.on_clicked(func)

    plt.show()
def add_filter_to_cams(file):
    for i in range(len(file)):
        file.loc[i, "Camera"] = f'{file.loc[i, "Filter"]}-{file.loc[i, "Camera"]}'
    return file
def start():

    galaxy_active = pd.read_csv(path+"/active_galaxies.csv")
    files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    for f in tqdm(files):  
        if f != "light_curves/661431822098-light-curves.csv" and f != "light_curves/661431908458-light-curves.csv" and f != "light_curves/42949788551-light-curves.csv": # fitler for specific galaxy
            #continue
            pass
        if f == "light_curves/name_id.csv" or f == "light_curves/.DS_Store"or f == path+"active_galaxies.csv":
            continue
        global current_ID
        current_ID = int(f.split("/")[-1].split("-")[0])
        global current_cuts
        current_cuts = 0
        global plot_cuts        
        plot_cuts = pd.DataFrame(columns = ["start","cam","shift"])
        plot_cuts = plot_cuts.astype({'start': 'datetime64[ns]', "cam": "str"})

        # Nur Galaien mit gewisser Aktivität laden
        if current_ID in galaxy_active["ID"].values:
            if (galaxy_active.loc[galaxy_active["ID"] == current_ID, "acivity"].values[0] <= treshold_F_var) and (galaxy_active.loc[galaxy_active["ID"] == current_ID, "R"].values[0] <= treshold_R):
                
                pass
            if galaxy_active.loc[galaxy_active["ID"] == current_ID, "ID"].values[0] not in galaxy_cuts:
                continue
            print(f"currentID: {current_ID}")
        # ========================================
            
        file = read_data_from_jd(f)
        file["JD"] = pd.to_datetime(file['JD'], origin='julian', unit='D')
        file['Camera'].replace('', np.nan)
        file.dropna(subset=['Camera'], inplace=True)
        file = file[(file["Flux Error"] < 1) & (file["Flux Error"] > 0)]
        file = file[(file["Mag Error"] < 1) & (file["Mag Error"] > 0)]
        file.reset_index(drop=True, inplace=True)
        file = add_filter_to_cams(file.copy()) # so dass filter vernachlässigt werden können 
        file.reset_index(drop=True, inplace=True)
        #print(f"\nGalaxy ID: {f}, name: {get_galaxy_name()} mit {file["Filter"].unique()} Filtern und {file["Camera"].unique()} Kameras")
        visualize1(file)
        #visualize_with_checkbuttons(file)


start()



# TODO: Lichtkurven importieren mit Koordinaten

#ESO 253 G003