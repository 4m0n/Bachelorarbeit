import pandas as pd

data = pd.read_csv("activity_curves/new_active_galaxies.csv")
data = data["name"].dropna()

data.to_csv("activity_curves/temp.txt", index=False, header=False)




print(data)