import pandas as pd
from os import listdir
import os
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg') 
import numpy as np
from prompt_toolkit import prompt
from matplotlib.widgets import TextBox, Button
import datetime
from astropy.time import Time

load_path = "final_light_curves/"
active_path = "activity_curves/"
value = "Flux"

kategorien = {
    0: "unbekannt",
    1: "Kategorie 1",
    2: "Kategorie 2",
    3: "Kategorie 3",
    4: "Kategorie 4",
    5: "Kategorie 5",
}

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
    def Datetime_in_Unix(date):
        unix = []
        for i in date:
            unix.append(i.timestamp())
        return unix
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
    def galaxie_list(data):
        data = pd.read_csv("galaxienotes.csv")
        
        index_list = pd.unique(data["category"])
        index_list = reversed(np.sort(index_list))
        
        data["fullname"] = ""
        for i in range(len(data)):
            newname = str(data["newname"].iloc[i])
            prefix = str(data["prefix"].iloc[i])
            name = str(data["name"].iloc[i])
            
            if len(newname) > 1 and newname != "nan":
                if len(prefix) > 1 and prefix != "nan":
                    data.loc[i, "fullname"] = f'{prefix}-{newname}'
                else:
                    data.loc[i, "fullname"] = newname
            elif len(prefix) > 1 and prefix != "nan":
                data.loc[i, "fullname"] = f'{prefix}-{name}'
            else:
                data.loc[i, "fullname"] = name
                
        #data["fullname"] = data["prefix"] + "-" + data["newname"]
        data['fullname'] = data['fullname'].fillna(data['name'])
        data["fullname"] = data["fullname"].str.lower()
        data.sort_values(by = "fullname", ascending=True, inplace = True)
        
        string = ""
        string2 = []
        for i in index_list:
            if i == -1:
                string += f"==== Ohne Kat. ====\n"
                string2.append("==== Ohne Kat. ====")
            else:
                string += f"==== Kat: {i} ====\n"
                string2.append(f"==== Kat: {i} ====")
            filtered_data = data[data["category"] == i]
            # Hinzufügen der 'fullname'-Werte zur Zeichenkette
            for fullname in filtered_data["fullname"].values:
                string += f"{fullname}\n"
                string2.append(fullname)
        #string2 = np.array(string2)
        maxlen = len(max(string2, key=len))
        breite = 3

        dim2 = len(string2)%breite
        dim2=breite - dim2


        for i in range(dim2):
            string2.append(" ")
        for i in range(len(string2)):
            diff = maxlen-len(string2[i])
            string2[i] = string2[i] + " "*diff


        dim2 = len(string2)/breite
        string2 = np.array(string2)

        string2 = string2.reshape(breite,int(dim2)).T
        text = "\n".join([" ".join(row) for row in string2])

        text = text.replace("][", "\n ")
        with open("Galaxie_Liste.txt", "w") as file:
            file.write(text)
           
class Plots:
    def show_plots(sortname = pd.DataFrame(),skip = "None", cat = 0):
        files = [f for f in listdir(load_path) if isfile(join(load_path, f))]
        files = [f.replace(".csv","") for f in files]
        if len(sortname) > 0:
            name_order = {name: i for i, name in enumerate(sortname)}
            sorted_names = sorted(files, key=lambda x: name_order.get(x, float('inf')))  # Standardwert für fehlende Namen
            files = sorted_names
            # alphabetisch sortieren
            files = [f for f in files if f in (sortname.values).tolist()]

        files.sort()

        # nach neuem Namen sortieren
        order = pd.read_csv("galaxienotes.csv")
        order = order[order["name"].isin(files)]
        order["fullname"] = ""
        for i in range(len(order)):
            newname = str(order["newname"].iloc[i])
            prefix = str(order["prefix"].iloc[i])
            name = str(order["name"].iloc[i])

            if len(newname) > 1 and newname != "nan":
                if len(prefix) > 1 and prefix != "nan":
                    order.loc[i, "fullname"] = f'{prefix}-{newname}'
                else:
                    order.loc[i, "fullname"] = newname
            elif len(prefix) > 1 and prefix != "nan":
                order.loc[i, "fullname"] = f'{prefix}-{name}'
            else:
                order.loc[i, "fullname"] = name
        #order["fullname"] = order["prefix"] + "-" + order["newname"]
        order['fullname'] = order['fullname'].fillna(order['name'])
        order["fullname"] = order["fullname"].replace(" ","")
        order["fullname"] = order["fullname"].str.lower()
        order.dropna(subset = ["name"], inplace = True)
        order.sort_values(by = "fullname", ascending=True, inplace = True)
        files = order["name"].values.tolist()

        if cat > 0:
            order = pd.read_csv("new_active_galaxies.csv")
            order = order[order["name"].isin(files)]
            order.sort_values(by = "activity", ascending=False, inplace = True)
            files = order["name"].values.tolist()
            files2 = []
            for val in files:
                files2.append(str(val)+".csv")
        s = True
        
        
        #print("\n\nSortierung")
        for file in files:
            if skip != "None":
                if file == skip:
                    s = False
                if s: continue
            name = file.replace(" ","")
            #print(f"Name: {order.loc[order['name'] == name, 'name'].values[0]} Wert: {order.loc[order['name'] == name, 'activity'].values[0]}")
            #Plots.plot_curves(file)
            Plots.plot_curves(file)
    
    def plot_curves2(name):
        print(f"name : {name}")
        file2 = FileManager.load_data(name)
        #file = BasicCalcs.normalize_null(file2)
        file = file2.copy()
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
        # farben = ["black","black","black","black","black","black","black",
        #           "black","black","black","black","black","black","black",
        #           "black","black","black","black","black","black","black",
        #           "black","black","black","black","black","black","black",
        #           "black","black","black","black","black","black","black"]
        
        cameras = cam.unique()  
        c2 = []
        for i in cam:
            c2.append(farben[np.where(cameras == i)[0][0]])
        print("jo3")
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

        liste = pd.read_csv("sortedcurves.csv")
        def remove_numbers(value):
            if isinstance(value, str):  # Prüfen, ob der Wert ein String ist
                if value[0] == "(":
                    return value[1:6]
            return value

        # Anwenden der Funktion auf alle Zellen
        print("jo1")
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
                #liste2.iloc[row, col] = remove_numbers(liste.iloc[row, col])
                pass
        
        liste2 = liste2.drop(columns = ["True_Count"])
        plt.title(f"NGC 4593",fontsize = 22)

        #plt.plot(x1,y1,zorder=10, label="30 Tage", color = "red")
        size = 24
        plt.scatter(x_1,y_1,c = c3, alpha=0.4, zorder=5, marker = "o",s = size) # plot verschobene orginalpunkte
        plt.scatter(x_2,y_2,c = c4, alpha=0.4, zorder=5, marker = "o",s = size) # plot verschobene orginalpunkte

        
        plt.grid(color='grey', linestyle='--', linewidth=0.5)
        plt.xlabel("Time [years]", fontsize=18)
        plt.ylabel("Flux [mJy]", fontsize=18)
        plt.legend(fontsize=10, frameon=True, fancybox=True, framealpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_curves(name):
    
        file = FileManager.load_data(name)
        
        if False:
            # Normal Fluss
            file = BasicCalcs.normalize_null(file)
#            file[value] = -file[value]
        else:
            # Relative Mag
            file[value] = 2.5 * np.log10(file[value])
            file = BasicCalcs.normalize_null(file)
            file[value] = file[value]/min(file[value])
            a = file[value].copy()
            b = file[value].copy()
                            
        file2 = BasicCalcs.rolling_mid(file.copy(),"30D")
        x, y, cam = file.index.copy(), file[value].copy(), file["Camera"].copy()
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

        liste = pd.read_csv("sortedcurves.csv")
        def remove_numbers(value):
            if isinstance(value, str):  # Prüfen, ob der Wert ein String ist
                if value[0] == "(":
                    return value[1:6]
            return value

        # Anwenden der Funktion auf alle Zellen
        
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
        def submit(event):
            global rvalue1,rvalue2,prefix1,newname1
            gnotes = pd.read_csv("galaxienotes.csv")
            notes = str(gnotes.loc[gnotes['name'] == name, "notes"].values[0])
            category = str(gnotes.loc[gnotes['name'] == name, "category"].values[0])
            prefix1 = str(gnotes.loc[gnotes['name'] == name, "prefix"].values[0])
            newname1 = str(gnotes.loc[gnotes['name'] == name, "newname"].values[0])
            if notes == "nan":
                notes = ""
            if category == -1:
                category = ""
            if prefix1 == "nan":
                prefix1 = ""
            if newname1 == "nan":
                newname1 = ""
            try:
                rvalue1 = float(x_textbox.text)
            except:
                rvalue1 = category
            if type(rvalue1) == type("string"):
                rvalue1 = -1
                
            rvalue2 = y_textbox.text
            prefix1 = x1_textbox.text
            newname1 = y2_textbox.text
            gnotes.loc[gnotes['name'] == name, "category"] = rvalue1
            gnotes.loc[gnotes['name'] == name, "notes"] = str(rvalue2)
            gnotes.loc[gnotes['name'] == name, "prefix"] = str(prefix1)
            gnotes.loc[gnotes['name'] == name, "newname"] = str(newname1)
            gnotes.to_csv("galaxienotes.csv", index=False)
            
            BasicCalcs.galaxie_list(gnotes)
            
            
        def next(event):
            submit(event)
            plt.close()
        def submit_exit(event):
            submit(event)
            exit()

        def speichern(event):
            x = np.concatenate((x_1, x_2))
            y = np.concatenate((y_1, y_2))
            #x = x.astype('datetime64[s]').astype('O')
            #x = BasicCalcs.Datetime_in_Unix(x)
            mjd_list = []

            for dt in x:
                time = Time(dt)
                mjd_list.append(time.mjd)
            x = mjd_list
            speicherdata = pd.DataFrame({'time': x, 'flux': y})
            speicherdata.to_csv(f'{name}_liste.csv', index=False)
            print(f"\ngespeichert als: {name}_liste.csv\n")

        gnotes = pd.read_csv("galaxienotes.csv")
        notes = str(gnotes.loc[gnotes['name'] == name, "notes"].values[0])
        category = str(gnotes.loc[gnotes['name'] == name, "category"].values[0])
        prefix = str(gnotes.loc[gnotes['name'] == name, "prefix"].values[0])   
        newname = str(gnotes.loc[gnotes['name'] == name, "newname"].values[0])
        if notes == "nan":
            notes = ""
        if float(category) == -1:
            category = ""
        if prefix == "nan":
            prefix = ""
        if newname == "nan":
            newname = ""
        fig, ax = plt.subplots()
        #plt.subplots_adjust(bottom=0.2,top=0.85)
        plt.subplots_adjust(bottom=0.2)

        axbox1 = fig.add_axes([0.12, 0.1, 0.1, 0.05])
        x_textbox = TextBox(axbox1, f"Category:", initial = category)
        
        axbox11 = fig.add_axes([0.30, 0.1, 0.1, 0.05])
        x1_textbox = TextBox(axbox11, f"Prefix:", initial = prefix)
        
        
        axbox2 = fig.add_axes([0.12, 0.05, 0.5, 0.05])
        y_textbox = TextBox(axbox2, f"Note:", initial = notes)

        axbox22 = fig.add_axes([0.12, 0.00, 0.4, 0.05])
        y2_textbox = TextBox(axbox22, f"Name:", initial = newname)

        axboxprint = fig.add_axes([0.63, 0.05, 0.08, 0.075])
        print_button = Button(axboxprint, "print")

        axbox3 = fig.add_axes([0.73, 0.05, 0.08, 0.075])
        submit_button = Button(axbox3, "Save")

        axbox33 = fig.add_axes([0.82, 0.05, 0.08, 0.075])
        submit_button2 = Button(axbox33, "Next")        
                
        axbox4 = fig.add_axes([0.91, 0.05, 0.06, 0.075])
        exit_button = Button(axbox4, "Exit")
        
        submit_button.on_clicked(submit)
        submit_button2.on_clicked(next)
        exit_button.on_clicked(submit_exit)
        print_button.on_clicked(speichern)
        def titleName(name,liste2, prefix, newname):
            name2 = name
            if newname != "":
                name2 = newname
            if prefix != "":
                name2 = f"{prefix}-{name2}"
            eigenschaften = liste2.loc[liste2['name'] == name, liste2.columns != 'name'].to_string(index=False)
            
            if newname == "": 
                return f"{name2}"
            else:
                return f"New: {name2}\nOld:{name}" 
        t = titleName(name,liste2, prefix, newname)
        #ax.set_title(f"Galaxy: {name}\n{liste2.loc[liste2['name'] == name, liste2.columns != 'name'].to_string(index=False)}")

        ax.set_title(t)
        ax.plot(x1, y1, zorder=10, label="30 Tage", color="red")
        ax.scatter(x_1, y_1, c=c3, alpha=0.4, zorder=5, marker="x")  # plot verschobene orginalpunkte
        ax.scatter(x_2, y_2, c=c4, alpha=0.4, zorder=5, marker="o")  # plot verschobene orginalpunkte

        ax.grid(color='grey', linestyle='--', linewidth=0.5)
        ax.set_xlabel("Zeit", fontsize=12)
        ax.set_ylabel("Fluss (normiert auf 1)", fontsize=12)
        ax.legend(fontsize=10, frameon=True, fancybox=True, framealpha=0.7)
        plt.show()
        
        return 0        
        
if True:
    # Fragezeichen entfernen
    directory = "final_light_curves/"
    for filename in os.listdir(directory):
        new_filename = filename.replace("?", "")
        if new_filename != filename:
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
            print(f'Renamed: {filename} -> {new_filename}')

    activegal = pd.read_csv("new_active_galaxies.csv")
    activegal["name"] = activegal["name"].str.replace("?", "", regex=False)
    activegal.to_csv("new_active_galaxies.csv", index=False)

    liste = listdir("final_light_curves")
    df_sorted = pd.read_csv("sortedcurves.csv")
    df_sorted["name"].replace("?", "", inplace=True)
    df_sorted["name"] = df_sorted["name"].str.replace("?", "", regex=False)
    df_sorted.to_csv("sortedcurves.csv", index=False)
    galaxienotes = pd.DataFrame({
        'name': df_sorted['name'],
        "category": [-1] * len(df_sorted),
        'notes': [None] * len(df_sorted),
        'prefix': [""] * len(df_sorted),
        'newname': [""] * len(df_sorted)
    }) 
    if not os.path.exists("galaxienotes.csv"):
        # DataFrame in die Datei schreiben
        galaxienotes.to_csv("galaxienotes.csv", index=False)
    data = pd.read_csv("galaxienotes.csv")
    # df erweitern um unbekannte einträge
    existing_names = set(data['name'])
    unique_notes = galaxienotes[~galaxienotes['name'].isin(existing_names)]
    data = pd.concat([data, unique_notes], ignore_index=True)


    if data.shape[1] == 3:
        # Eine dritte Spalte mit -1 hinzufügen
        data['category'] = -1
        data["prefix"] = ""
        data["newname"] = ""
        # DataFrame in die Datei zurückschreiben
        data.to_csv("galaxienotes.csv", index=False)
    
    # sort galaxienotes for name
    #data = pd.read_csv("galaxienotes.csv")
    data["fullname"] = ""
    for i in range(len(data)):
        newname = str(data["newname"].iloc[i])
        prefix = str(data["prefix"].iloc[i])
        name = str(data["name"].iloc[i])
        
        if len(newname) > 1 and newname != "nan":
            if len(prefix) > 1 and prefix != "nan":
                data.loc[i, "fullname"] = f'{prefix}-{newname}'
            else:
                data.loc[i, "fullname"] = newname
        elif len(prefix) > 1 and prefix != "nan":
            data.loc[i, "fullname"] = f'{prefix}-{name}'
        else:
            data.loc[i, "fullname"] = name
            
    #data["fullname"] = f'{data["prefix"]}-{data["newname"]}'
    data['fullname'] = data['fullname'].fillna(data['name'])

    #data["fullname"].replace(" ","", inplace=True)
    data["fullname"] = data["fullname"].str.replace(" ", "", regex=False)
    data["fullname"] = data["fullname"].str.lower()
    data.sort_values(by = "fullname", ascending=True, inplace = True)

    data.drop(columns = ["fullname"], inplace = True)
    data.to_csv("galaxienotes.csv", index=False)

            
        
        
        
    print(f"\n\nWo soll es losgehen? \n1 - Ganz am Anfang \n2 - Bei der ersten Galaxie ohne Kategorie\n3 - Für Galaxie auswahl mit Name\n4 - Für Kategorie [?] [?]\n\n")
    number = input(f"\n\nZahl: ")
    if number == "1":
        print("\n\n======= Galaxienotizen =======\n\n")
        Plots.show_plots(df_sorted["name"])
    elif number == "2":
        print("\n\n======= Galaxienotizen =======\n\n")
        data = pd.read_csv("galaxienotes.csv")
        #first_empty_note = data[data['notes'].isna()].iloc[0]
        first_empty_cat = data[data['category'] == -1].iloc[0]
        print(first_empty_cat['name'])
        Plots.show_plots(df_sorted["name"],first_empty_cat['name'])
    elif number == "3":
        name = str(input(f"\n\nName der Galaxie (aus der Liste kopieren): "))
        print("\n\n======= Galaxienotizen =======\n\n")  
        galaxienotes = pd.read_csv("galaxienotes.csv")
        if name in galaxienotes["newname"].values:
            name = galaxienotes.loc[galaxienotes['newname'] == name, 'name'].values[0]
        Plots.show_plots(df_sorted["name"],name)
    elif number == "4":
        print("\nZuerst Kategorie eingeben, danach muss angegeben werden, wie mit der Kategorie vorgegangen werden soll.\n")
        cat = str(input(f"\nCategory: \n"))
        print("\nMögliche Optionen:\n = um Nur diese Kategorie anzuzeigen\n!= um alle ausser diese Kategorie anzuzeigen\n>= um nur diese und größere Kategorien anzuzeigen\n")
        zeichen = str(input(f"=,!=,>=: "))
        data = pd.read_csv("galaxienotes.csv")
        if zeichen == "=":
            show = data[data['category'] == float(cat)]
            if len(show)<=0:
                print("Keine Galaxien vorhanden mit diesen Kriterien")
                exit()
            print(f"show: {show["name"]}")
            Plots.show_plots(show["name"], cat = 1)
        elif zeichen == "!=":
            show = data[(data['category'] != float(cat)) & (data['category'] >= 0)]
            if len(show)<=0:
                print("Keine Galaxien vorhanden mit diesen Kriterien")
                exit()
            Plots.show_plots(show["name"], cat = 1)
        elif zeichen == ">=":
            show = data[data['category'] >= float(cat)]
            if len(show)<=0:
                print("Keine Galaxien vorhanden mit diesen Kriterien")
                exit()
            Plots.show_plots(show["name"], cat = 1)

        
        
    else:
        print("\n\n Falscher Input, Neustarten...\n\n")
        exit()
        
# CGCG 49-155 = SDSS J15242+0451