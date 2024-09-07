import pandas as pd
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
import math
path = "light_curves/"
#path = "light_curves_specific/"
value = "Flux"
mid = 13

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
def remove_outliers_mad(file, threshold=3):
    def neumann_diff(a,b, threshold=3): # format von a: pandas [times, values]
        # normieren der Lichtkurven -> mean = 0, std = 1
        mean_a = a["value"].mean()
        std_a = a["value"].std()

        mean_b = b["value"].mean()
        std_b = b["value"].std()

        t_min = abs(a["JD"].min() - b["JD"].max()).days
        t_max = abs(a["JD"].max() - b["JD"].min()).days

        if t_min > 500: # sonst dauert es zu lange
            t_min = 500
        if t_max > 500:
            t_max = 500

        print(f"t_min: {t_min} t_max: {t_max}")
        a["value"] = (a["value"]-mean_a)/std_a
        b["value"] = (b["value"]-mean_b)/std_b
        # Daten einsortieren
        F = pd.concat([a, b])
        F = F.sort_values(by='JD').reset_index(drop=True)

        # um tau verschobene Daten
        T = []
        r = range(-t_min,t_max)
        for tau in r:
            c = b.copy()
            c["JD"] = c["JD"] + pd.Timedelta(days=tau)
            F_tau = pd.concat([a, c])
            F_tau = F_tau.sort_values(by='JD').reset_index(drop=True)

            #T(tau) berechnen
            T.append(0)
            for i in range(len(F)-1):
                T[-1] += (F["value"][i] - F_tau["value"][i+1])**2
            T[-1] = T[-1] * 1/(len(F)-1)
        print(f"Kleinste Tau ist: {r[T.index(min(T))]}")

        #plotten



        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        c = b.copy()
        c["JD"] = c["JD"] + pd.Timedelta(days=r[T.index(min(T))])
        # Plot 1: r vs T
        ax1.plot(r, T)
        ax1.set_title('Plot 1: r vs T')
        ax1.set_xlabel('r')
        ax1.set_ylabel('T')

        # Plot 2: Zeit vs Value für DataFrames a, b und c
        ax2.plot(a['JD'], a['value'], label='a')
        ax2.plot(c['JD'], c['value'], label='new')
        ax2.plot(b['JD'], b['value'], label='original', color='black', alpha=0.5)
        ax2.set_title('Plot 2: Time vs Value for a, b, and c')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Value')
        ax2.legend()  # Legende hinzufügen

        # Layout anpassen und anzeigen
        plt.tight_layout()
        plt.show()

        return r[T.index(min(T))]

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
                a = file.loc[(file[filter] == x), ['JD', value]].copy()
                a.rename(columns={value: 'value'}, inplace=True)
                b = file.loc[(file[filter] == y), ['JD', value]].copy()
                b.rename(columns={value: 'value'}, inplace=True)
                neumann_diff(a,b)

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
    move("Camera",cameras)

    return file

# Beispiel: Entfernen von Ausreißern aus einer Spalte 'Flux'
def rolling_mid(file):
    file.set_index('JD', inplace=True)
    file.sort_index(inplace=True)
    # Berechne den gleitenden Mittelwert über 28 Tage
    file[value] = file[value].rolling(window='28D').mean()
    file.reset_index(inplace=True)
    return file

def visualize1(file):
    x_org, y_org, cam = file["JD"].copy(), file[value].copy(), file["Camera"].copy()
    file = remove_outliers_mad(file)
    x1,y1 = file["JD"].copy(), file[value].copy()
    file = rolling_mid(file)
    x2,y2 = file["JD"].copy(), file[value].copy() # "JD" ist jetzt der Index!!!

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
        
    plt.plot(x2,y2,zorder=4)   
    plt.scatter(x2,y2,marker = "x",zorder=4)  
    plt.scatter(x1,y1,c = c,zorder=3)
    plt.scatter(x_org,y_org,c = "black", alpha=0.4, zorder=1)
    plt.scatter(x_org,y_org,c = c2, alpha=0.4, zorder=5, marker = "x")
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
        file.reset_index(drop=True, inplace=True)
        print(f"Galaxy ID: {f}")
        visualize1(file)


start()




# verschiedene Kameras kombinieren wie Filter
# rolling mean für Zeitintervall um durchschnittswert zu erhalten

# zuerst noise entfernen, dann rolling mean