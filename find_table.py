import requests
import pandas as pd
from bs4 import BeautifulSoup

url = "http://asas-sn.ifa.hawaii.edu/skypatrol/objects/661430607288"

# open Website
response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table')
    df = pd.read_html(str(table))[0]
    df.to_csv('tabelle.csv', index=False)
    print("Tabelle erfolgreich gespeichert.")
else:
    print(f"Fehler beim Abrufen der Website: {response.status_code}")