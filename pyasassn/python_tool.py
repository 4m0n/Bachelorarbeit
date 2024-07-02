import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pyasassn.client import SkyPatrolClient
import requests


filename = '100targets.txt'

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
    #test = client.cone_search(ra_deg=270, dec_deg=88, radius=4, catalog='aavsovsx')
    #test = client.query_list(my_tic_ids, catalog = "stellar_main", id_col = "tic_id", download=True)
    print("\ncheck 1\n")
    test = client.cone_search('18:54:11.5',' -88:02:55.22' ,radius =2.0 ,units='deg',download=True)
    print("\ncheck 2\n")
    print(f"Test: \n{test} \n\n")

def test():
    client = SkyPatrolClient()

    # Funktion zum Abfragen der Lichtkurven für eine bestimmte Position (RA, DEC) und Radius
    def get_light_curves_by_coordinates(ra_deg, dec_deg, radius):
        result = client.cone_search(ra_deg=ra_deg, dec_deg=dec_deg, radius=radius, catalog='master_list')
        if not result.empty:
            print(f"Found {len(result)} objects.")
            for idx, row in result.iterrows():
                asas_sn_id = row['asas_sn_id']
                # Korrektur: Erstellung eines query_hash basierend auf den Abfrageparametern
                query_params = {
                    'ra_deg': ra_deg,
                    'dec_deg': dec_deg,
                    'radius': radius,
                    'catalog': 'asas_sn',
                    'id': asas_sn_id
                }
                # Generiere den query_hash (vereinfacht als String-Kombination)
                query_hash = hash(frozenset(query_params.items()))
                # Abrufen der Lichtkurven
                light_curve = client._get_curves(tar_ids=[asas_sn_id], catalog='asas_sn', save_dir=None, file_format='json', query_hash=query_hash)
                print(f"Light Curve for ASAS-SN ID {asas_sn_id}:\n{light_curve}")
        else:
            print("No objects found in the given coordinates.")

    # Funktion zum Abfragen der Lichtkurven für bestimmte Namen (z.B., ASASSN-V J182608.32-864925.1)
    def get_light_curves_by_name(name):
        result = client.query_list(name, catalog='aavsovsx', id_col='name')
        if not result.empty:
            asas_sn_id = result['asas_sn_id'].iloc[0]
            # Korrektur: Erstellung eines query_hash basierend auf den Abfrageparametern
            query_params = {
                'name': name,
                'catalog': 'asas_sn',
                'id': asas_sn_id
            }
            # Generiere den query_hash (vereinfacht als String-Kombination)
            query_hash = hash(frozenset(query_params.items()))
            # Abrufen der Lichtkurven
            light_curve = client._get_curves(tar_ids=[asas_sn_id], catalog='asas_sn', save_dir=None, file_format='json', query_hash=query_hash)
            print(f"Light Curve for {name}:\n{light_curve}")
        else:
            print("No objects found with the given name.")

    # Beispiel für die Verwendung
    ra_deg = 270.0  # Rechte Ascension in Grad
    dec_deg = 88.0  # Deklination in Grad
    radius = 4.0  # Radius des Kegelsuchbereichs in Bogenminuten

    # Abfrage nach Koordinaten
    #get_light_curves_by_coordinates(ra_deg, dec_deg, radius)

    # Abfrage nach Namen
    name = 'ASASSN-V J182608.32-864925.1'
    get_light_curves_by_name(name)
    


#============================
    

    #TimeoutError: Lightcurve servers unavailable, try again later


#============================




if __name__ == '__main__':
    #data = load_data(filename)
    #print(f"data: \n{data}\n")
    #load_galaxys(data)
    test()
    














