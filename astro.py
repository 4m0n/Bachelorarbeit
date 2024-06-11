import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import optimize
from datetime import datetime, timedelta
import requests
import time
from selenium import webdriver 
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import pandas as pd

def input(elem, text):
    elem.clear()
    elem.send_keys(text)
    return
def get_data(data):
    for name in data["name            "]:
        name = name.strip()
        driver = webdriver.Firefox()  
        wait = WebDriverWait(driver, 10)
        driver.get("https://asas-sn.osu.edu")
        driver.fullscreen_window()
        # SET GALAXY NAME
        galaxy_name = wait.until(EC.presence_of_element_located((By.ID, "starParamForJQ")))
        input(galaxy_name, name) 
        print("entered Name")
        # CLICK RESOLVE BUTTON
        commit_button = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'button.btn.btn-sm.btn-primary#sesameName')))
        commit_button.click()
        print("Clicked resolve button")
        # SET OBS TIME
        obs_time = wait.until(EC.presence_of_element_located((By.NAME, "query[duration]")))
         
        print("entered time") 
        # DO CAPTCHA
        commit_button = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'recaptcha-checkbox-border')))
        commit_button.click()
        input(obs_time, 10)
        # START QUERY 
        commit_button = wait.until(EC.presence_of_element_located((By.NAME, "commit")))
        commit_button.submit()
        time.sleep(10)
        # CLOSE
        driver.quit()
        break 





data = pd.read_csv("BrowseTargets.csv", delimiter='|', header=0, skipinitialspace=True, )
#print(data)
get_data(data)