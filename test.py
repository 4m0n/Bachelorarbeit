import numpy as np
import matplotlib.pyplot as plt



if False: #normieren der bedingun
    daten = np.loadtxt("light_curves/active_galaxies_normiert.csv", delimiter=',', skiprows=1, usecols=-1)
    normiert = np.loadtxt("light_curves/active_galaxies.csv", delimiter=',', skiprows=1, usecols=-1)

    #a * x = y -> a = y / x
    norm = []
    x = []
    for i in range(len(daten)):
        norm.append(daten[i]/normiert[i])
        x.append(i)
            
    plt.scatter(x,norm,color = "red")
    plt.show()

if True:# Korellation F_var und R
    x = np.loadtxt("light_curves/active_galaxies.csv", delimiter=',', skiprows=1,usecols=-3) #F_var
    y = np.loadtxt("light_curves/active_galaxies.csv", delimiter=',', skiprows=1,usecols=-2) #R
    times = np.loadtxt("light_curves/active_galaxies.csv", delimiter=',', skiprows=1,usecols=-1)
    plt.hlines(1.2,min(x),max(x)) # R
    plt.vlines(0.5,min(y),max(y)) # F
    plt.scatter(x,y,color = "red")
    plt.xlabel("F_var")
    plt.ylabel("R")
    plt.grid()
    plt.show()
    plt.close()
