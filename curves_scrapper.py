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

def start():
    def input(elem, text):
        elem.clear()
        elem.send_keys(text)
        return


    #df = pd.read_csv('pyasassn_tool/pd_100targets.csv',nrows=5)
    df = pd.read_csv('pyasassn_tool/pd_1000targets.csv',nrows=1000)
    df_loaded = pd.DataFrame(columns=df.columns)
    loaded = np.loadtxt("light_curves/name_id.csv",dtype=str, skiprows=1,delimiter=",")
    ra = np.loadtxt("light_curves/name_id.csv",dtype=str, skiprows=2,delimiter=",")
    dec = np.loadtxt("light_curves/name_id.csv",dtype=str, skiprows=3,delimiter=",")
    print(f"loaded: \n{loaded}\n\ndf: \n{df}\n\n")
    for val in tqdm(reversed(loaded)): # check if data already exists
        index = 5
        index = df.loc[df["name"]==val[1]].index
        if len(index)>=1:
            df.drop(index = index, inplace=True)
            df.reset_index(drop=True, inplace=True)
            ra = np.delete(ra,index)
            dec = np.delete(dec,index)  
            

    # Erstellen eines neuen DataFrame mit nur der Spalte "name" und einer neuen Spalte "ID"
    df = pd.DataFrame({
        'name': df['name'],
        'ID': None, 
        'ra': ra,
        'dec': dec
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
                    df.at[index, 'ID'] = 0
                else:
                    print(f"Error: {error}")
            except:
                pass
            else:
                break



        if skip:
            continue
        try:
            # === Choose Dataset ===
            #button = wait.until(EC.visibility_of_element_located((By.XPATH, "//a[@rel='noopener noreferrer']")))
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
            df.at[index, 'ID'] = 0
            continue


    if len(loaded) > 1:
        df.to_csv('light_curves/name_id.csv', mode='a', header=False)
    else: 
        df.to_csv('light_curves/name_id.csv') 




    time.sleep(5)
    driver.quit()

    print("done")


    """
    - überlapp berechnen 
    - 
    """
start()