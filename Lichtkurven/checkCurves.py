import pandas as pd
from os import listdir
import os
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg') 
import numpy as np
from prompt_toolkit import prompt

load_path = "final_light_curves/"
active_path = "activity_curves/"
value = "Flux"
class FileManager:
    def load_data(file):
        if os.path.exists(load_path + file + ".csv"):
            data = pd.read_csv(load_path + file + ".csv", index_col = 0)
            #print(f"Test:\nIndex:\n{data.index}\nFlux:\n{data["Flux"]}\nFlux Error:\n{data["Flux Error"]}\nCamera\n{data["Camera"]}")
            data.index = pd.to_datetime(data.index)
            return data    
        return pd.DataFrame()
    def load_cuts(name):
        return 0
        #activity = pd.read_csv(activity_path)
        if name in activity["name"].values:
            cuts = activity.loc[activity['name'] == name, 'cuts'].values[0]
        else: cuts = -1
        return cuts
class BasicCalcs:
    def normalize_null(file):
        curve = file.copy()
        curve[value] = curve[value] - curve[value].min() 
        curve[value] = curve[value]/curve[value].max()
        return curve
    def rolling_mid(file, rolling_time ="28D"):
        file2 = file.copy() 
        file2.sort_index(inplace=True)
        file2[value] = file2[value].rolling(window=rolling_time, center = True, min_periods = 4).mean()
        return file2
           
class Plots:
    def show_plots(sortname = pd.DataFrame(),skip = "None"):
        files = [f for f in listdir(load_path) if isfile(join(load_path, f))]
        if len(sortname) > 0:
            name_order = {name: i for i, name in enumerate(sortname+".csv")}
            sorted_names = sorted(files, key=lambda x: name_order.get(x, float('inf')))  # Standardwert für fehlende Namen
            files = sorted_names
        
        
        s = True
        for file in files:
            print(file)
            if skip != "None":
                if file[:-4] == skip:
                    s = False
                if s: continue
            name = file[:-4].replace(" ","")
            Plots.plot_curves(file[:-4])
            
    def plot_curves(name):

        file2 = FileManager.load_data(name)
        file = BasicCalcs.normalize_null(file2)
        x, y, cam = file.index.copy(), file[value].copy(), file["Camera"].copy()
        file2 = BasicCalcs.rolling_mid(file,"30D")
        x1, y1 = file2.index.copy(), file2[value].copy()
        # Kamera farben
        
        farben = [
            "blue", "black", "magenta", "orange", "purple", 
            "brown", "gray", "olive", "darkblue", "lime", 
            "indigo", "gold", "darkgreen", "teal", "red", 
            "maroon", "navy", "darkred", "forestgreen", 
            "slategray", "darkslateblue", "chocolate", "darkorange", 
            "seagreen", "sienna", "darkmagenta", "midnightblue", 
            "firebrick", "cadetblue", "dodgerblue", "peru", 
            "rosybrown", "saddlebrown", "darkolivegreen", "steelblue", 
            "tomato", "mediumblue", "deepskyblue", "crimson", "mediumvioletred", 
            "orchid", "plum", "slateblue", "turquoise", "violet", "darkcyan", 
            "darkorchid", "mediumorchid"
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

        # params = FindActive.load_parameters(name)
        # if name in galaxy_active["name"].values:
        #     if not CONDITION(**params):
        #         #plt.title(f"NICHT VARIABEL Galaxy: {name}, cuts: {cuts} \nTH F: {F_threshold} Activity F: {tr_F_var}\nTR R: {R_threshold} Activity R: {tr_R}\nTR Amp: {amp_diff_threshold} Amp: {tr_amplitude}, T = {round(T/86400)} Jahre")
        #         plt.title(f"NICHT VARIABEL Galaxy: {name}")
        #     else:
        #         #plt.title(f"VARIABEL \nGalaxy: {name}, cuts: {cuts} \nTH F: {F_threshold} Activity F: {tr_F_var}\nTR R: {R_threshold} Activity R: {tr_R}\n TR Amp: {amp_diff_threshold} Amp: {tr_amplitude}, T = {round(T/86400)} Jahre")
        #         plt.title(f"VARIABEL \nGalaxy: {name}")
        # else:
        #     plt.title(f"Galaxy: {name} - nicht gefunden")
        liste = pd.read_csv("sortedcurves.csv")
        def remove_numbers(value):
            if isinstance(value, str):  # Prüfen, ob der Wert ein String ist
                if value[0] == "(":
                    return value[1:6]
            return value

        # Anwenden der Funktion auf alle Zellen
        #liste2 = liste.map(remove_numbers)
        
        def remove_numbers(value):
            if isinstance(value, str):  # Prüfen, ob der Wert ein String ist
                if value[0] == "(":
                    return value[1:6]
            return value

        # Leeren DataFrame mit gleicher Struktur erstellen
        liste2 = pd.DataFrame(index=liste.index, columns=liste.columns)

        # Durch jede Zelle des DataFrames iterieren
        for row in range(liste.shape[0]):  # Zeilenweise
            for col in range(liste.shape[1]):  # Spaltenweise
                liste2.iloc[row, col] = remove_numbers(liste.iloc[row, col])
        
        
        liste2 = liste2.drop(columns = ["True_Count"])

        plt.title(f"Galaxy: {name}\n{liste2.loc[liste2['name'] == name, liste2.columns != 'name'].to_string(index=False)}")

        plt.plot(x1,y1,zorder=10, label="30 Tage", color = "red")
        plt.scatter(x_1,y_1,c = c3, alpha=0.4, zorder=5, marker = "x") # plot verschobene orginalpunkte
        plt.scatter(x_2,y_2,c = c4, alpha=0.4, zorder=5, marker = "o") # plot verschobene orginalpunkte
  
        
        plt.grid(color='grey', linestyle='--', linewidth=0.5)
        plt.xlabel("Zeit", fontsize=12)
        plt.ylabel("Fluss (normiert auf 1)", fontsize=12)
        plt.legend(fontsize=10, frameon=True, fancybox=True, framealpha=0.7)
        plt.tight_layout()
        plt.show()
        
        gnotes = pd.read_csv("galaxienotes.csv")
        notes = str(gnotes.loc[gnotes['name'] == name, "notes"].values[0])
        if notes == "nan":
            notes = ""
        notes = prompt(f"Galaxie {name} Notizen: ", default=notes)
        gnotes.loc[gnotes['name'] == name, "notes"] = notes
        gnotes.to_csv("galaxienotes.csv", index=False)

        plt.close()
        
        
        
        
if True:
    liste = listdir("final_light_curves")
    df_sorted = pd.read_csv("sortedcurves.csv")

    galaxienotes = pd.DataFrame({
        'name': df_sorted['name'],
        'notes': [None] * len(df_sorted)
    }) 
    if not os.path.exists("galaxienotes.csv"):
        # DataFrame in die Datei schreiben
        galaxienotes.to_csv("galaxienotes.csv", index=False)
    print(f"\n\nWo soll es losgehen? \n1 - Ganz am Anfang \n2 - Bei der ersten Galaxie ohne Beschriftung\n3 - Für Galaxie auswahl\n\n")
    number = input(f"\n\nZahl: ")
    if number == "1":
        print("\n\n======= Galaxienotizen =======\n\n")
        Plots.show_plots(df_sorted["name"])
    elif number == "2":
        print("\n\n======= Galaxienotizen =======\n\n")
        data = pd.read_csv("galaxienotes.csv")
        first_empty_note = data[data['notes'].isna()].iloc[0]
        print(first_empty_note['name'])
        Plots.show_plots(df_sorted["name"],first_empty_note['name'])
    elif number == "3":
        name = str(input(f"\n\nName der Galaxie (aus der Liste kopieren): "))
        print("\n\n======= Galaxienotizen =======\n\n")        
        Plots.show_plots(df_sorted["name"],name)
    else:
        print("\n\n Falscher Input, Neustarten...\n\n")
        exit()