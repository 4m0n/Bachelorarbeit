import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



a = [1,2,3,4,5,6,7,8,9,10]
b = [4,5,6,7,8,92,10,11,12,13]

print(np.corrcoef(a,b),np.corrcoef(a,b)[0,1])


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

if False:# Korellation F_var und R
    x = np.loadtxt(path, delimiter=',', skiprows=1,usecols=1) #F_var
    y = np.loadtxt(path, delimiter=',', skiprows=1,usecols=2) #R
    z = np.loadtxt(path, delimiter=',', skiprows=1,usecols=4) #cuts
    times = np.loadtxt(path, delimiter=',', skiprows=1,usecols=3)
    name = np.loadtxt(path, delimiter=',', skiprows=1,usecols=0,dtype=str)
    

        
    aktive_prozent = 0
    variabel = pd.DataFrame(columns=["name","F","R"])
    for i in range(len(y)):
        if y[i] > R_treshold and F_treshold < x[i]:
            aktive_prozent += 1
            if len(variabel) == 0:
                variabel = pd.DataFrame({"name":[name[i]],"F":[x[i]],"R":[y[i]]})
            variabel = pd.concat([variabel,pd.DataFrame({"name":[name[i]],"F":[x[i]],"R":[y[i]]})])
    aktive_prozent = round(aktive_prozent / len(y) * 10_000)/100

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
