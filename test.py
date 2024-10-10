import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

F_treshold = 0.0025
R_treshold = 1.2
path_new = "activity_curves/new_active_galaxies.csv"
path_old = "light_curves/active_galaxies.csv"
path = path_new
if False: #normieren der bedingun
    daten = np.loadtxt("light_curves/active_galaxies_normiert.csv", delimiter=',', skiprows=1, usecols=-1)
    normiert = np.loadtxt(path, delimiter=',', skiprows=1, usecols=-1)

    #a * x = y -> a = y / x
    norm = []
    x = []
    for i in range(len(daten)):
        norm.append(daten[i]/normiert[i])
        x.append(i)
            
    plt.scatter(x,norm,color = "red")
    plt.show()

if True:# Korellation F_var und R
    x = np.loadtxt(path, delimiter=',', skiprows=1,usecols=1) #F_var
    y = np.loadtxt(path, delimiter=',', skiprows=1,usecols=2) #R
    z = np.loadtxt(path, delimiter=',', skiprows=1,usecols=4) #cuts
    times = np.loadtxt(path, delimiter=',', skiprows=1,usecols=3)
    name = np.loadtxt(path, delimiter=',', skiprows=1,usecols=0,dtype=str)
    

        
    aktive_prozent = 0
    variabel = pd.DataFrame(columns=["name","F","R"])
    for i in range(len(y)):
        if y[i] > R_treshold or F_treshold < x[i]:
            aktive_prozent += 1
            variabel = pd.concat([variabel,pd.DataFrame({"name":[name[i]],"F":[x[i]],"R":[y[i]]})])
    aktive_prozent = round(aktive_prozent / len(y) * 10000)/100
    print(f"Variable Galaxien nach aktuellem Filter:\n{variabel.to_string()}")
    
    print(f"min y {min(y)} max y {max(y)}")
    plt.hlines(R_treshold,min(x),max(x)) # R
    plt.vlines(F_treshold,min(y),max(y)) # F
    plt.scatter(x,y,c = z,s = 50, edgecolor = "k", alpha = 0.5)
    plt.colorbar(label='Z-Wert (Heatmap)')
    plt.xlabel("F_var")
    plt.ylabel("R")
    plt.title(f"Anzahl: {len(x)}, Variabel: {aktive_prozent}%")
    plt.grid()
    plt.show()
    plt.close()
