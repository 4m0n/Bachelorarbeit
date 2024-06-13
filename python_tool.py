import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pyasassn.client import SkyPatrolClient

filename = '1000targets.txt'

def load_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    headers = lines[2].strip().split('|')
    headers = [header.strip() for header in headers]

    data = []
    for line in lines[3:]:
        values = line.strip().split('|')
        values = [value.strip() for value in values]
        for i in range(len(values)):
            if values[i] == "":
                continue
            if values[i] == "":
                values[i] = 0

        data.append(dict(zip(headers, values)))

    df = pd.DataFrame(data)
    df['app_mag'] = pd.to_numeric(df['app_mag'], errors='coerce')
    df['redshift'] = pd.to_numeric(df['redshift'], errors='coerce')
    df.to_csv("pd_" + filename[:-4] + ".csv", index=False)
    return df
def plot():
    matplotlib.use('Qt5Agg')
    print(data)
    print("PLOT:")
    plt.scatter(data['app_mag'],data['redshift'], s = 0.5)
    plt.xlabel("redshift")
    plt.ylabel("app_mag")

    plt.savefig('plot.png',dpi = 1200)
    print("done")
def load_galaxys(data):
    client = SkyPatrolClient()
    my_tic_ids = [6658326,46783395,1021890,2798421]
    test = client.cone_search(ra_deg=270, dec_deg=88, radius=4, catalog='aavsovsx')
    print(test)
    test = client.query_list(my_tic_ids, catalog = "stellar_main", id_col = "tic_id")
    test = client.random_sample(1000, catalog = "morx")
    print(test)
data = load_data(filename)
load_galaxys(data)














