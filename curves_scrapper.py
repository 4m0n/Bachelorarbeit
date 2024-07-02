import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import optimize
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from fpdf import FPDF
import requests
import time
from selenium import webdriver 
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
import os
import pandas as pd
import json
import re




def input(elem, text):
    elem.clear()
    elem.send_keys(text)
    return


df = pd.read_csv('pyasassn/pd_100targets.csv',nrows=5)

# Erstellen eines neuen DataFrame mit nur der Spalte "name" und einer neuen Spalte "ID"
df = pd.DataFrame({
    'name': df['name'],
    'ID': None  # Hier kannst du später die ID-Werte hinzufügen
})


driver = webdriver.Firefox()  
wait = WebDriverWait(driver, 100)
driver.get("http://asas-sn.ifa.hawaii.edu/skypatrol/")
driver.fullscreen_window()

# === Select SIMBAD Resolver ===
#noopener noreferrer
original_window = driver.current_window_handle
button = wait.until(EC.visibility_of_element_located((By.XPATH, "//button[@value='simbadSearch']")))
button.click()

for index, row in df.iterrows():
    name = row['name']
    print(f"searching for {name}    {index+1}:{len(df)}")
    # === Select Galaxy ===
    galaxy = wait.until(EC.presence_of_element_located((By.NAME, "simbadSearchName")))
    input(galaxy, name)
    # ===  Search ===
    search = wait.until(EC.presence_of_element_located((By.NAME, "submit")))
    search.click()
    # === Choose Dataset ===
    button = wait.until(EC.visibility_of_element_located((By.XPATH, "//a[@rel='noopener noreferrer']")))
    button.click()

    # === Switch Tabs === 
    windows = driver.window_handles

    # Zum neuen Fenster wechseln
    for window in windows:
        if window != original_window:
            driver.switch_to.window(window)
            break
    # === Save ID ===
    element = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'h3.page-title')))
    text = element.text
    # Die Nummer extrahieren (angenommen, die Nummer ist die erste Zahl im Text)
    match = re.search(r'ID: (\d+)', text)
    if match:
        number = match.group(1)
        print(f"Gefundene Nummer: {number}\n")
    else:
        print("Keine Nummer gefunden")
        number = 0
    
    df.at[index, 'ID'] = number
    
    # === Download csv ===
    button = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'button.form-btn')))
    button.click()
    driver.close()
    driver.switch_to.window(original_window)


df.to_csv('name_id.csv')




time.sleep(5)
driver.quit()

print("done")

