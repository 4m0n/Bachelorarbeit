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
import requests
import bs4 as bs

from rich import print
from rich.traceback import install
from rich.console import Console
install()
console = Console() 


data = {
  'rightAscension': "00:42:44.3",
  'declination': '41:16:10',
}

response = requests.get("http://asas-sn.ifa.hawaii.edu/skypatrol/")

console.print(response.headers)
print("\n<=======================================>\n")

response = requests.post(f'http://asas-sn.ifa.hawaii.edu/skypatrol/', data=data)




console.print(response.headers,response.text)
















