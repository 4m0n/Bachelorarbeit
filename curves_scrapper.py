import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import optimize
from datetime import datetime, timedelta
import time
from selenium import webdriver 
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
import pandas as pd
import re
import csv
from tqdm import tqdm


def calc_degrees(ra,dec):
    ra_h = float(ra[0:2])
    ra_min = float(ra[3:5])
    ra_sec = float(ra[6:])
    ra = float(15*ra_h + (ra_min/60 *15)+(ra_sec/3600 *15))
    
    
    if dec[0] == "+" or dec[0] == "-":
        vorzeichen = int(dec[0] +"1")
        dec = dec[1:]
    dec_min = float(dec[3:5])
    dec_sec = float(dec[6:])
    dec = float(dec[0:2]) + (dec_min/60) + (dec_sec/3600)*vorzeichen
    print(ra,dec)
    return ra,dec


calc_degrees("00 42 44.3","+41 16 10")


def start():
    def input(elem, text):
        elem.clear()
        elem.send_keys(text)
        return


    #df = pd.read_csv('pyasassn_tool/pd_100targets.csv',nrows=5)
    df = pd.read_csv('pyasassn_tool/pd_1000targets copy.csv',nrows=1000)
    df_loaded = pd.DataFrame(columns=df.columns)
    loaded = np.loadtxt("light_curves_new1/name_id.csv",dtype=str, skiprows=1,delimiter=",")

    print(f"loaded: \n{loaded}\n\ndf: \n{df}\n\n")
    for val in tqdm(reversed(loaded)): # check if data already exists
        index = 5
        index = df.loc[df["name"]==val[1]].index
        if len(index)>=1:
            df.drop(index = index, inplace=True)
            df.reset_index(drop=True, inplace=True)

            

    # Erstellen eines neuen DataFrame mit nur der Spalte "name" und einer neuen Spalte "ID"
    df = pd.DataFrame({
        'name': df['name'],
        'ID': None, 
        'ra': df['ra'],
        'dec': df['dec']
    })
    
 
    driver = webdriver.Firefox()  
    wait = WebDriverWait(driver, 100)
    wait2 = WebDriverWait(driver, 1)
    driver.get("http://asas-sn.ifa.hawaii.edu/skypatrol/")
    driver.fullscreen_window()

    # === Select SIMBAD Resolver ===
    #noopener noreferrer
    original_window = driver.current_window_handle
    button = wait.until(EC.visibility_of_element_located((By.XPATH, "//button[@value='simbadSearch']")))
    button.click()
    for index, row in tqdm(df.iterrows()):
        skip = False
        try:
            driver.switch_to.window(original_window)
            button = wait.until(EC.visibility_of_element_located((By.XPATH, "//button[@value='simbadSearch']")))
            button.click()
        except:
            continue
            
        try:
            name = row['name']
            print(f"searching for {name}    {index+1}:{len(df)}")
            # === Select Galaxy ===
            galaxy = wait.until(EC.presence_of_element_located((By.NAME, "simbadSearchName")))
            input(galaxy, name)
            # ===  Search ===
            search = wait.until(EC.presence_of_element_located((By.NAME, "submit")))
            search.click()
            # === Check if exists === 
            time.sleep(2)
        except:
            continue

        for i in range(100):
            try:
                print("1")
                button = wait2.until(EC.visibility_of_element_located((By.XPATH, "//a[@rel='noopener noreferrer']")))
            except:
                pass
            else:
                break
            try:
                print("2")
                error = wait2.until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'p.error-message'))).text
                if error == "Your search has rendered no results.":
                    skip = True
                    
                else:
                    print(f"Error: {error}")
            except:
                pass
            else:
                break



        if skip:
            try:
                ra = row['ra']
                dec = row['dec']
                ra,dec = calc_degrees(ra,dec)
                radius = 7.8
                print("\n\n Test search coordinates\n\n")
                button = wait.until(EC.visibility_of_element_located((By.XPATH, "//button[@value='coneSearch']")))
                button.click()
                galaxy = wait.until(EC.presence_of_element_located((By.NAME, "rightAscension")))
                input(galaxy, ra)
                galaxy = wait.until(EC.presence_of_element_located((By.NAME, "declination")))
                input(galaxy, dec)
                galaxy = wait.until(EC.presence_of_element_located((By.NAME, "radius")))
                input(galaxy, radius)
                # ===  Search ===
                search = wait.until(EC.presence_of_element_located((By.NAME, "submit")))
                search.click()
                # === Check if exists === 
                time.sleep(2)
            except:
                print("couldnt find coordinates")
                continue
        
        
        
        try:
            # === Choose Dataset ===
            #button = wait.until(EC.visibility_of_element_located((By.XPATH, "//a[@rel='noopener noreferrer']")))
            button = wait2.until(EC.visibility_of_element_located((By.XPATH, "//a[@rel='noopener noreferrer']")))
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
        except:
            print("Error: No data found")
            df.at[index, 'ID'] = -1
            continue


    if len(loaded) > 1:
        df.to_csv('light_curves_new1/name_id.csv', mode='a', header=False)
    else: 
        df.to_csv('light_curves_new1/name_id.csv') 




    time.sleep(5)
    driver.quit()

    print("done")


    """
    - überlapp berechnen 
    - 
    """
start()