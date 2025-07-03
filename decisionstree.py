from sklearn import datasets, model_selection, neighbors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as  np
from tqdm import tqdm
from itertools import combinations
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

control = False

# data prep
data_X = pd.read_csv("activity_curves/new_active_galaxies.csv")
if control: # Daten von meinem Alg
    data_y = pd.read_csv("sortedcurves.csv")
    data_y.rename(columns={"Unnamed: 0": "name"}, inplace=True)
    data_y["variable"] = data_y["True_Count"] >= 2
    data_y = data_y[["name", "variable"]]
else: # Daten von Wolfram
    data_y = pd.read_csv("Lichtkurven/galaxienotes.csv")
    data_y["variable"] = data_y["category"] >= 2
    data_y = data_y[["name", "variable"]]


data_y.replace({"variable": {True: 1, False: 0}}, inplace=True)

#data_y = data_y[["name", "variable"]]
data = data_X.merge(data_y, on="name", how="inner")
data_y = data[["variable"]]
#data['periodic'] = data['periodic'].apply(lambda x: float(x.split(",")[1][:-1]))

# data.drop(columns=['R', 'activity*R', 'cuts', 'amplitude', 'amp_diff',
#        'period', 'periodicpercent', 'Dt', 'std', 'up', 'down', 'mean', 'peakA',
#        'peakC', 'pointCount', 'StartEndDiff', 'redshift',
#        'periodicFastFreq', 'magnitude'], inplace=True)

data.drop(columns=["variable"],inplace = True)

data = data.infer_objects(copy=False)
data_X = data.copy()
#data_X.drop(columns=["name","variable","cuts","StartEndDiff","magnitude","redshift","up","down","activity*R","Dt"], inplace=True)
data_X.drop(columns=["name"], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=42)
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()
# ==============


def knn(X,y,plot=False, grid_search=False):
    if grid_search:
        param_grid = {
        'n_neighbors': [2,5,10,20,50,100]
        }

        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
        grid_search.fit(X_train, y_train)

        print("Beste Parameter:", grid_search.best_params_)
    
    
    rnd_clf = KNeighborsClassifier(n_neighbors=20) # this command (with its params) comes from the documentation
    rnd_clf.fit(X, y)
    
    if plot:
        y_pred = rnd_clf.predict(X_test)

        # Berechnung der Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['False', 'True'], yticklabels=['False', 'True'])
        plt.xlabel('Vorhergesagte Werte', size=14)
        plt.ylabel('Echte Werte', size=14)
        plt.title('Confusion Matrix', size=16)
        plt.show()
    return rnd_clf


def dec_tree(X,y, plot=False, grid_search=False):

    rnd_clf = DecisionTreeClassifier(random_state=42) # this command (with its params) comes from the documentation
    rnd_clf.fit(X, y)
    
    if plot:
        feature_importance = rnd_clf.feature_importances_
        sorted_idx = feature_importance.argsort()  
        plt.barh(X.columns.values[sorted_idx], feature_importance[sorted_idx])
        plt.xlabel('feature importance')
        plt.show()
        
        
        y_pred = rnd_clf.predict(X_test)

        # Berechnung der Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['False', 'True'], yticklabels=['False', 'True'])
        plt.xlabel('Vorhergesagte Werte', size=14)
        plt.ylabel('Echte Werte', size=14)
        plt.title('Confusion Matrix', size=16)
        plt.show()
    return rnd_clf

def rand_forest(X, y, plot=False, grid_search=False):
    from sklearn.ensemble import RandomForestClassifier
    
    if grid_search:
        param_grid = {
        'n_estimators': [100, 200, 300,800,1200],
        'max_leaf_nodes': [5,10, 20, 30,80],
        'random_state': [42]
        }

        grid_search = GridSearchCV(RandomForestClassifier(n_jobs=-1), param_grid, cv=5)
        grid_search.fit(X_train, y_train)

        print("Beste Parameter:", grid_search.best_params_)
    
    rnd_clf = RandomForestClassifier(n_estimators=100, max_leaf_nodes=30, n_jobs=-1, random_state=42)  # this command (with its params) comes from the documentation
    rnd_clf.fit(X, y)
    
    if plot:
        feature_importance = rnd_clf.feature_importances_
        sorted_idx = feature_importance.argsort()  
        plt.barh(X.columns.values[sorted_idx], feature_importance[sorted_idx])
        plt.xlabel('feature importance')
        plt.show()


        # Vorhersagen des Modells auf den Testdaten
        y_pred = rnd_clf.predict(X_test)

        # Berechnung der Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['False', 'True'], yticklabels=['False', 'True'])
        plt.xlabel('Vorhergesagte Werte', size=14)
        plt.ylabel('Echte Werte', size=14)
        plt.title('Confusion Matrix', size=16)
        plt.show()
    return rnd_clf


def contour_plot(X, y):
    #model = dec_tree(X, y) 
    model = rand_forest(X, y) 
    model = knn(X, y) 

    from matplotlib.colors import ListedColormap
    cm_tree = ListedColormap(['#e58139', '#399de5'])  # colors are chosen to coincide with decision tree

    # Konvertiere X in ein NumPy-Array, falls es ein DataFrame ist
    if isinstance(X, pd.DataFrame):
        X = X.values

    h = .001  # step size in the mesh

    plt.figure(figsize=(7, 7))

    # determine boundaries
    x1_min, x1_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    x2_min, x2_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    # assign decision tree predictions to each mesh point
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # plot training data
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_tree, edgecolors='k')

    # plot decision boundary
    plt.contourf(xx, yy, Z, cmap=cm_tree, alpha=.1)

    plt.xlabel('feature $x_0$', size=16)
    plt.ylabel('feature $x_1$', size=16)

    plt.show()
    
    



neigh = knn(X_train, y_train, True,True)
print(f"\nTrain Score: {neigh.score(X_train, y_train)}\nTest Score: {neigh.score(X_test, y_test)}\n")

contour_plot(X_train[["amplitude","periodicFast"]], y_train)