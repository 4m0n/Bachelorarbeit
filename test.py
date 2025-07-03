import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('activity_curves/new_active_galaxies.csv')


def secondToYears(value):
    return value / (365 * 24 * 60 * 60)
#liste = secondToYears(data["period"].values).tolist()
liste = data["periodicFast"]
array = []
for val in liste:
    array.append(float(val[1:-1]))

print(f"min: {min(array)}, max: {max(array)} median: {np.median(array)}")
exit()
def load(filepath):
    with open(filepath, 'r') as file:
        # Suche die Zeile, die mit "JD" beginnt
        for i, line in enumerate(file):
            if line.startswith("JD"):
                start_line = i
                break

    # Lies die Datei ab der gefundenen Zeile ein
    df = pd.read_csv(filepath, skiprows=range(start_line))
    return df

data = load("light_curves/661431908458-light-curves.csv")
#data["JD"] = pd.to_datetime(data['JD'], origin='julian', unit='D')

print(data["JD"],"\n\n",data["Flux"])

plt.scatter(data["JD"],data["Flux"])
plt.show()


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
