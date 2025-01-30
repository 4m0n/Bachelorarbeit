from matplotlib.widgets import TextBox, Button

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
from scipy import optimize
import yaml
import scipy.signal as signal
from astropy.timeseries import LombScargle
from scipy.interpolate import UnivariateSpline


from rich import print
from rich.traceback import install
from rich.console import Console
install()
console = Console()


# Create figure
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
# I used pandas to read data from a csv file
# but in this case I will just use dummy values as example
X = [];Y1 = [];Y2 = [];Y3 = []
for i in range(10):
    X.append(i)
    Y1.append(i)
    Y2.append(i*2)
    Y3.append(i**2)
# Plot the data
ax.plot(X,Y1,X,Y2,X,Y3)
# Handles submit button
def submit(event):
    print("yes")
    print("x =", x_textbox.text)
    print("y =", y_textbox.text)
    x = int(x_textbox.text)
    y = y_textbox.text
    X = [];Y1 = [];Y2 = [];Y3 = []
    if x not in ["", None]:
        for i in range(x):
            X.append(i)
            Y1.append(i-1)
            Y2.append(i*3)
            Y3.append(i**3)
        #fig.pop(0)

        ax.plot(X,Y1,X,Y2,X,Y3)
# Text box to input x value
axbox1 = fig.add_axes([0.1, 0.1, 0.5, 0.05])
x_textbox = TextBox(axbox1, "New X")
# Text box to input y value
axbox2 = fig.add_axes([0.1, 0.05, 0.5, 0.05])
y_textbox = TextBox(axbox2, "New Y")
# Submit button
axbox3 = fig.add_axes([0.81, 0.05, 0.1, 0.075])
submit_button = Button(axbox3, "Submit!")
submit_button.on_clicked(submit)
plt.show()

