import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load(filepath,fertig = False):
    with open(filepath, 'r') as file:
        # Suche die Zeile, die mit "JD" beginnt
        for i, line in enumerate(file):
            if line.startswith("JD"):
                start_line = i
                break

    # Lies die Datei ab der gefundenen Zeile ein

    df = pd.read_csv(filepath, skiprows=range(start_line))
    if fertig == False:
        df["JD"] = pd.to_datetime(df['JD'], origin='julian', unit='D')
    else:
        df["JD"] = pd.to_datetime(df['JD'])
    df.set_index("JD", inplace=True)
    print(df)
    return df
def plot_curves(name, fertig = False):
    farben = ["blue", "black", "magenta", "orange", "purple",
    "brown", "gray", "olive", "darkblue", "lime",
    "indigo", "gold", "darkgreen", "teal", "red",
    "maroon", "navy", "darkred", "forestgreen",
    "slategray", "darkslateblue", "chocolate", "darkorange",
    "seagreen", "sienna", "darkmagenta", "midnightblue",
    "firebrick", "cadetblue", "dodgerblue", "peru",
    "rosybrown", "saddlebrown", "darkolivegreen", "steelblue",
    "tomato", "mediumblue", "deepskyblue", "crimson", "mediumvioletred",
    "orchid", "plum", "slateblue", "turquoise", "violet", "darkcyan",
    "darkorchid", "mediumorchid"]

    value = "Flux"
    file = load(name,fertig)
    x, y, cam = file.index.copy(), file[value].copy(), file["Camera"].copy()
    error1 = file[f"{value} Error"].copy()

    # Kamera farben
    cameras = cam.unique()
    c2 = []
    for i in cam:
        c2.append(farben[np.where(cameras == i)[0][0]])

    c3 = []
    c4 = []
    for index, i in enumerate(cam):
        if file["Filter"].iloc[index] == "V":
            c3.append(farben[np.where(cameras == i)[0][0]])
        else:
            c4.append(farben[np.where(cameras == i)[0][0]])

    x_1 = file.loc[file["Filter"] == "V", :].index.copy()
    x_2 = file.loc[file["Filter"] == "g", :].index.copy()
    y_1 = file.loc[file["Filter"] == "V", value].values.copy()
    y_2 = file.loc[file["Filter"] == "g", value].values.copy()
    error_1 = file.loc[file["Filter"] == "V", f"{value} Error"].values.copy()
    error_2 = file.loc[file["Filter"] == "g", f"{value} Error"].values.copy()


    plt.scatter(x_1, y_1, c=c3, alpha=0.4, zorder=5, marker="x")  # plot verschobene orginalpunkte
    plt.scatter(x_2, y_2, c=c4, alpha=0.4, zorder=5, marker="o")  # plot verschobene orginalpunkte

    # plt.errorbar(x_1, y_1, yerr=error_1, alpha=0.4, zorder=5, label='V', ecolor=c3, linestyle="None", barsabove=True,fmt="")
    # plt.errorbar(x_2, y_2, yerr=error_2, alpha=0.4, zorder=5, label="g",ecolor=c4, linestyle="None")

    # x_temp,y_temp,_,_ = FindActive.peak(name)

    # ==== Standartabweichung ====

    # plt.hlines(y.mean()+y.std(),min(x),max(x), label = "Mittelwert", color = "black",alpha = 0.5)
    # plt.hlines(y.mean()-y.std(),min(x),max(x), label = "Mittelwert", color = "black",alpha = 0.5)

    # rolling_std = y.rolling(window="30D", center = False, min_periods = 8).std()
    # rolling_mid = y.rolling(window="30D", center = False, min_periods = 8).mean()
    # plt.plot(x,rolling_mid+rolling_std, label = "Standartabweichung", color = "black",alpha = 0.5,zorder = 6)
    # plt.plot(x,rolling_mid-rolling_std, label = "Standartabweichung", color = "black",alpha = 0.5,zorder = 6)

    # ============================

    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.xlabel("Zeit", fontsize=12)
    plt.ylabel("Fluss", fontsize=12)
    plt.legend(fontsize=10, frameon=True, fancybox=True, framealpha=0.7)
    plt.tight_layout()
    plt.show()
    plt.close()



plot_curves("light_curves/661431908458-light-curves.csv")
plot_curves("final_light_curves/NGC4593.csv", True)


