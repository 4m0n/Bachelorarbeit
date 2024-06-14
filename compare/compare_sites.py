import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord
from astropy import units as u
from scipy.optimize import curve_fit

size = 10
alpha = 0.4
location = EarthLocation(0, 0, 0)
y1, y2 = 'Flux (mJy)','Flux' # oder "Mag" bei beiden
y1, y2 = 'Mag','Mag'
amplitude = [('Flux (mJy)','Flux'),('Mag','Mag')]


def plot_graphs(name, y1, y2):
    plt.scatter(sp1["HJD"], sp1[y1].apply(float_conv), color = "red", alpha=alpha, s=size, marker = "x", label = "SkyPatrol V1")
    plt.scatter(sp21["JD"], sp21[y2].apply(float_conv), color = "blue", alpha=alpha, s=size, marker = "x", label = "SkyPatrol V2")
    plt.legend(loc = "best")
    plt.grid(which='major', color='grey', linestyle='--', alpha=0.75)
    plt.title(name)
    #plt.scatter(sp22["JD"], sp22[y2].apply(float_conv), color = "green", alpha=alpha, s=size)
    plt.savefig(name+".png", dpi = 2000)
    plt.show()

def float_conv(x):
    return float(x)

def HJD_to_JD(ra, dec, location, data):
    # Standort der Beobachtung
    sky_coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    # Konvertierung HJD zu JD
    hjd_times = Time(data['HJD'].values, format='jd', scale='utc', location=location)
    jd_times = hjd_times.jd - hjd_times.light_travel_time(sky_coord).value
    # Erstellen eines neuen DataFrames mit JD Werten
    new_data = data.copy()
    new_data['JD'] = jd_times
    return new_data

def JD_to_HJD(ra, dec, location, data):
    # Standort der Beobachtung
    sky_coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    # Konvertierung JD zu HJD
    jd_times = Time(data['JD'].values, format='jd', scale='utc', location=location)
    hjd_times = jd_times.jd + jd_times.light_travel_time(sky_coord).value
    # Erstellen eines neuen DataFrames mit HJD Werten
    new_data = data.copy()
    new_data['HJD'] = hjd_times
    return new_data
    
for y1, y2 in amplitude:
    sp1 = pd.read_csv("skypatrol1_flux.csv")
    sp21 = pd.read_csv("sp2_661430329816.csv")
    sp22 = pd.read_csv("sp2_661430607288.csv")

    # === SETUP DATA ===

    def filter1(data, min_date):
        filter_cam = data["Filter"] == "g"
        data = data.loc[filter_cam]
        filter_cam = data["JD"] >= min_date
        data = data.loc[filter_cam]
        return data

    sp1 = HJD_to_JD(10.684844167300014, 41.26908712360001,location,sp1)
    min_date = sp1["HJD"].min()

    sp1 = filter1(sp1, min_date)
    sp21 = filter1(sp21, min_date)
    sp22 = filter1(sp22, min_date)

    # === Plot before === 
    plot_graphs("orignal"+y2,y1,y2)

    # === CALCULATE DIFF ===

    def linear_function(x, A, B):
        return A + B * x
    def move(x, a,b):
        return a + x*b


    if False: # simple linear
        med1 = sp1[sp1[y1] < 11][y2].mean()
        med21 = sp21[sp21[y2] < 11][y2].mean()    
        med22 = sp22[sp22[y2] < 11][y2].mean()
        print(med1,med21,med22)

        sp21[y2] = sp21[y2] + (med1-med21)
        sp22[y2] = sp22[y2] + (med1-med22)
    else: # shift + scale factor
        interpolated_y2 = np.interp(sp1['HJD'], sp21['JD'], sp21[y2])
        popt, pcov = curve_fit(linear_function, sp1[y1], interpolated_y2)
        A, B = popt
        sp21[y2] = sp21[y2].apply(lambda x: (x-A)/B)
        print(f"Optimale Parameter: A = {A}, B = {B}")

    plot_graphs("new"+y2,y1,y2)

    """plt.scatter(sp1["HJD"], sp1["Flux (mJy)"].apply(float_conv), color = "red", alpha=alpha, s=size)
    plt.scatter(sp21["JD"], sp21["Flux"].apply(float_conv), color = "blue", alpha=alpha, s=size)
    plt.scatter(sp22["JD"], sp22["Flux"].apply(float_conv), color = "green", alpha=alpha, s=size)
    plt.savefig("nr3.png", dpi = 1200)
    plt.show()"""

