import pandas as pd
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from numba import jit
from tqdm import tqdm
path = "light_curves/"
#path = "light_curves_specific/"
value = "Flux"
mid = 13


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
            ax1.set_title('Plot 1: r vs T')
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
        for _ in range(1):
            for c in ["V","g"]:
                df = file[file.Filter == c].reset_index(drop=True)
                
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
                    mean = df[value].rolling(window=mid, center=True).median() # median besser als mean?
                    mean_mean = mean.mean()
                    std = df[value].rolling(window=mid, center=True).std()
                    std_mean = std.mean()
                    full_std = df[value].std()
                    mean = np.where(np.isnan(mean), mean_mean, mean)
                    std = np.where(np.isnan(std), std_mean, std)

                    #print(f"mean: {mean} \n\nstd {std}")
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

def neumann_cam_shift(data,curve):

    #find overlapp 
    start_overlap = max(data["JD"].min(), curve["JD"].min())
    end_overlap = min(data["JD"].max(), curve["JD"].max())
    if end_overlap < start_overlap:
        return curve
    
    
    main_curve = data[(data['JD'] >= start_overlap) & (data['JD'] <= end_overlap)].copy()
    fit_curve = curve[(curve['JD'] >= start_overlap) & (curve['JD'] <= end_overlap)].copy()
    main_curve.reset_index(drop=True, inplace=True)
    fit_curve.reset_index(drop=True, inplace=True)
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
    
    if True: #plot zeigen
        print(f"\noptimaler shift bei {shift} (mean shift: {mean_diff} -> Diff: {shift-mean_diff} \nmit T min = {min(T)} bei index: {T.index(min(T))} bei R: {R[T.index(min(T))]}\n")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        # Plot 1: r vs T
        all_data = pd.concat([data,curve])
        all_data.sort_values(by='JD', inplace=True)
        fit_curve[value] = fit_curve_backup[value] + shift
        curve[value] = curve[value] + shift
        ax1.scatter(R, T)
        ax1.set_title('Plot 1: r vs T')
        ax1.set_xlabel('r')
        ax1.set_ylabel('T')
        ax1.vlines(shift, color='red', label='min T', ymin = min(T), ymax = max(T))
        ax1.grid(True)
        ax2.grid(True)
        ax2.plot(data['JD'], data[value], label='existing',alpha = 0.8, color = "green")
        ax2.plot(curve['JD'], curve[value], label='new',alpha = 0.8, color = "red")
        ax2.plot(mean_plot_curve['JD'], mean_plot_curve[value], label='MEAN rechnung',alpha = 0.8,color = "orange")
        ax2.plot(all_data['JD'], all_data[value], label='all data',alpha = 0.3,color = "blue")
        ax2.set_title(f"\noptimaler shift bei {shift}\n")
        ax2.set_xlabel('Time')
        ax2.set_ylabel('wert')
        ax2.legend()  # Legende hinzufügen

        # Layout anpassen und anzeigen
        plt.tight_layout()
        plt.show()
    return curve
def shift_cam_and_filters(file):
    cameras = file["Camera"].unique() 
    filters = file["Filter"].unique()
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
        print(f"cam: {cameras[i]} nr: {i}")
        fit_curve = file[file["Camera"] == cameras[i]].copy()
        # Check ob es größere Lücken oder Sprünge gibt -> Dann Lichtkurve weiter unterteilen
        
        
        fit_curve = neumann_cam_shift(main_curve,fit_curve)
        main_curve = pd.concat([main_curve, fit_curve], ignore_index=True)
        main_curve.sort_values(by='JD', inplace=True)
    
    return main_curve
    
# Beispiel: Entfernen von Ausreißern aus einer Spalte 'Flux'
def rolling_mid(file, timeslot ="28D"):
    file.set_index('JD', inplace=True)
    file.sort_index(inplace=True)
    # Berechne den gleitenden Mittelwert über 28 Tage
    file[value] = file[value].rolling(window=timeslot, center = False).mean().copy()
    file.reset_index(inplace=True)
    return file
@funcion_time
def visualize1(file):
    x_org, y_org, cam = file["JD"].copy(), file[value].copy(), file["Camera"].copy()
    file = remove_outliers_mad(file)
    file = shift_cam_and_filters(file)
    
    #file = neumann_remove(file)
    x1,y1 = file["JD"].copy(), file[value].copy()
    file2 = rolling_mid(file,"10D")
    x2,y2 = file2["JD"].copy(), file2[value].copy() # "JD" ist jetzt der Index!!!
    file2 = rolling_mid(file,"30D")
    x3,y3 = file2["JD"].copy(), file2[value].copy() # "JD" ist jetzt der Index!!!
    file2 = rolling_mid(file,"90D")
    x4,y4 = file2["JD"].copy(), file2[value].copy() # "JD" ist jetzt der Index!!!

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
    "gold", "darkgreen", "teal"
    ]
    cameras = cam.unique()  
    c2 = []
    for i in cam:
        c2.append(farben[np.where(cameras == i)[0][0]])

    # === Plot
        
    #plt.plot(x2,y2,zorder=4, label="10 Tage")
    plt.plot(x3,y3,zorder=6, label="30 Tage")
    #plt.plot(x4,y4,zorder=5, label="90 Tage")   
    #plt.scatter(x2,y2,marker = "x",zorder=4)  
    #plt.scatter(x1,y1,c = c,zorder=3, alpha=0.3)
    #plt.scatter(x_org,y_org,c = "black", alpha=0.4, zorder=1)
    plt.scatter(x_org,y_org,c = c2, alpha=0.4, zorder=5, marker = "x")
    plt.grid()
    plt.legend()
    plt.show()

def start():
    files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    for f in files:  
        if f == "light_curves/name_id.csv" or f == "light_curves/.DS_Store":
            continue
        file = read_data_from_jd(f)
        file["JD"] = pd.to_datetime(file['JD'], origin='julian', unit='D')
        file['Camera'].replace('', np.nan)
        file.dropna(subset=['Camera'], inplace=True)
        file = file[file["Flux Error"] < 1]
        file = file[file["Mag Error"] < 1]
        file.reset_index(drop=True, inplace=True)
        print(f"\nGalaxy ID: {f} mit {file["Filter"].unique()} Filtern und {file["Camera"].unique()} Kameras")
        visualize1(file)


start()




# verschiedene Kameras kombinieren wie Filter
# rolling mean für Zeitintervall um durchschnittswert zu erhalten

# zuerst noise entfernen, dann rolling mean