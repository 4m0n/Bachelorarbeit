import pandas as pd
from astropy.time import Time

from rich import print
from rich.traceback import install
from rich.console import Console
install()
console = Console()
name = "PG 0043+039"
path = f"final_light_curves/{name}.csv"

def read_data_from_jd(filepath):
    df = pd.read_csv(filepath, usecols=["JD", "Flux"])
    return df

file = read_data_from_jd(path) 

file['JD'] = pd.to_datetime(file['JD'])
file["Date"] = file["JD"]
file['JD'] = file['JD'].apply(lambda x: Time(x).jd)

console.print(file)

# speichere die datei in eine csv datei

file.to_csv(f"light_curves_new1/{name}.csv", index=False)