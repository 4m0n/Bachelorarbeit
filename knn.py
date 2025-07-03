from sklearn import datasets, model_selection, neighbors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as  np
from tqdm import tqdm
from itertools import combinations

# data prep
data_X = pd.read_csv("activity_curves/new_active_galaxies.csv")
data_y = pd.read_csv("sortedcurves.csv")
data_y2 = pd.read_csv("Lichtkurven/galaxienotes.csv")

#data_y["active"] = data_y["True_Count"] >= 2
data_y["variable"] = data_y2["category"] >= 2


data_y.replace({"variable": {True: 1, False: 0}}, inplace=True)
data_y.rename(columns={"Unnamed: 0": "name"}, inplace=True)
data_y = data_y[["name", "variable"]]
data = data_X.merge(data_y, on="name", how="inner")

data_X = data.copy()
data_X.drop(columns=["name","variable","cuts","StartEndDiff","magnitude","redshift","up","down","activity*R","Dt"], inplace=True)

data_y = data[["variable"]]

X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=42)
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()
# ==============




def train_knn(X,y,k=1):
    neigh = neighbors.KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None) # this command (with its params) comes from the documentation
    neigh.fit(X, y)
    return neigh




    


def find_best_k(X_train,y_train, X_test, y_test):
    train_score = []
    test_score = []
    for k in tqdm(range(1,140)):
        neigh = train_knn(X_train, y_train, k=k)
        train_score.append(neigh.score(X_train, y_train))
        test_score.append(neigh.score(X_test, y_test))
    print("Train scores:", train_score)
    print("Test scores:", test_score)
    plt.plot(range(1, len(train_score)+1), train_score, label="Train Score", color = "blue")
    plt.plot(range(1, len(test_score)+1), test_score, label="Test Score", color = "orange")
    #plt.xscale("log")
    plt.grid()
    plt.legend()
    plt.title("Train and Score Plot (Accuracy)")
    plt.xlabel("k")
    plt.ylabel("1-Accuracy")
    plt.show()

def comb_train(X_train,y_train, X_test, y_test):
    if True:
        columns = list(X_train.columns)
        all_combinations = []

        results = pd.DataFrame(columns=["train_score", "test_score", "combination"])    
        for r in range(1, len(columns) + 1):
            all_combinations.extend(combinations(columns, r))
            #comb = list(combinations(columns, r))[0]
        for comb in tqdm(all_combinations):    
            
            dataX = pd.DataFrame()
            data_testX = pd.DataFrame()
            for val in comb:
                dataX[val] = X_train[val]
                data_testX[val] = X_test[val]
            neigh = train_knn(dataX, y_train, k=20)
            
            new_row = pd.DataFrame({
                "train_score": [neigh.score(dataX, y_train)],
                "test_score": [neigh.score(data_testX, y_test)],
                "combination": [dataX.columns.tolist()]
            })

            
            results = pd.concat([results, new_row], ignore_index=True)
        results.to_csv("knn_test2.csv", index=False)
    else:
        results = pd.read_csv("knn_test2.csv")
    #results = results.sort_values(by="test_score", ascending=False)
    print(results)
    plt.plot(range(1, len(results["train_score"])+1), results["train_score"], label="Train Score", color = "blue")
    plt.plot(range(1, len(results["test_score"])+1), results["test_score"], label="Test Score", color = "orange")
    #plt.xscale("log")
    plt.grid()
    plt.legend()
    plt.title("Train and Score Plot (1-Accuracy)")
    plt.xlabel("k")
    plt.ylabel("1-Accuracy")
    plt.show()
    
    
    
comb_train(X_train,y_train, X_test, y_test)
#find_best_k(X_train,y_train, X_test, y_test)