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
from tqdm import tqdm

from rich import print
from rich.traceback import install
from rich.console import Console
install()
console = Console()

normalizeing = "true"

textsize1 = 1.9
textsize2 = 19

def load_data(control):
    # data prep
    data_X = pd.read_csv("activity_curves/new_active_galaxies.csv")
    if control: # Daten von meinem Alg
        print("Using data from my algorithm")
        data_y = pd.read_csv("sortedcurves.csv")
        data_y.rename(columns={"Unnamed: 0": "name"}, inplace=True)
        data_y["variable"] = data_y["True_Count"] >= 2
        data_y = data_y[["name", "variable"]]
        print(len(data_y[data_y["variable"] == True]))
    else: # Daten von Wolfram
        print("Using data from Wolfram")
        data_y = pd.read_csv("Lichtkurven/galaxienotes.csv")
        data_y["variable"] = data_y["category"] >= 3
        data_y = data_y[["name", "variable"]]
        print(len(data_y[data_y["variable"] == True]))


    #data_y.replace({"variable": {True: 1, False: 0}}, inplace=True)
    data_y["variable"] = data_y["variable"].map({True: 1, False: 0})
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

    return X_train,X_test,y_train,y_test

def knn(X,y,plot=False, grid_search=False,show_plot=True,title=""):
    if grid_search:
        param_grid = {
            'n_neighbors': [2, 5, 10, 20, 50, 100],
            'weights': ['uniform', 'distance'],
            'p': [1, 2],
            'metric': ['minkowski', 'euclidean', 'manhattan'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }

        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y)

        print("Beste Parameter:", grid_search.best_params_)
    
    
    rnd_clf = KNeighborsClassifier(n_neighbors=20) # this command (with its params) comes from the documentation
    rnd_clf.fit(X, y)
    
    if plot:
        y_pred = rnd_clf.predict(X)

        # Berechnung der Confusion Matrix
        cm = confusion_matrix(y, y_pred,normalize=normalizeing)

        # Plot-Einstellungen
        plt.figure(figsize=(6, 6))
        sns.set(font_scale=textsize1)  # Größere Schrift für bessere Lesbarkeit
        ax = sns.heatmap(
            cm,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=['False', 'True'],
            yticklabels=['False', 'True'],
            cbar=False,
            linewidths=0.5,
            linecolor='gray',
            square=True
        )
        ax.set_xlabel('Predicted Value', fontsize=textsize2)
        ax.set_ylabel('True Value', fontsize=textsize2)
        #ax.set_title('Confusion Matrix', fontsize=15, pad=15)
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"bilder/cn_matrix_knn{title}.png", dpi=400, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()
    return rnd_clf


def dec_tree(X,y, plot=False, grid_search=False,show_plot=True,title=""):

    if grid_search:
        param_grid = {
        'max_depth': [3, 9, 10, 11,None],  # Begrenzung der Baumtiefe
        'min_samples_split': [2, 5, 8,10,20,30,50],  # Mindestanzahl von Samples für einen Split
        'min_samples_leaf': [2, 5,10],  # Mindestanzahl von Samples pro Blatt
        'max_leaf_nodes': [None, 2,10, 20,50],  # Begrenzung der Anzahl der Blätter
        }

        # GridSearchCV initialisieren
        grid_search = GridSearchCV(
            estimator=DecisionTreeClassifier(random_state=42),
            param_grid=param_grid,
            scoring='accuracy',  # Bewertungsmetrik
            cv=5,  # Cross-Validation mit 5 Folds
            n_jobs=-1  # Parallelisierung
        )

        # GridSearch ausführen
        grid_search.fit(X, y)    
        print("Beste Parameter:", grid_search.best_params_)
        print("Beste Genauigkeit:", grid_search.best_score_)
        rnd_clf = grid_search.best_estimator_

    else:
        rnd_clf = DecisionTreeClassifier(
            max_depth=9,            # Begrenzung der Baumtiefe
            min_samples_split=30,    # Mindestanzahl von Samples für einen Split
            min_samples_leaf=2,     # Mindestanzahl von Samples pro Blatt
            max_leaf_nodes=10,    # Begrenzung der Anzahl der Blätter
            random_state=42         # Zufallszustand für Reproduzierbarkeit
        )

    rnd_clf.fit(X, y)
    
    if plot:
        feature_importance = rnd_clf.feature_importances_
        sorted_idx = feature_importance.argsort()  
        plt.barh(X.columns.values[sorted_idx], feature_importance[sorted_idx])
        plt.xlabel('feature importance')
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        
        y_pred = rnd_clf.predict(X)

        # Berechnung der Confusion Matrix
        cm = confusion_matrix(y, y_pred,normalize=normalizeing)

        # Plot-Einstellungen
        plt.figure(figsize=(6, 6))
        sns.set(font_scale=textsize1)  # Größere Schrift für bessere Lesbarkeit
        ax = sns.heatmap(
            cm,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=['False', 'True'],
            yticklabels=['False', 'True'],
            cbar=False,
            linewidths=0.5,
            linecolor='gray',
            square=True
        )
        ax.set_xlabel('Predicted Value', fontsize=textsize2)
        ax.set_ylabel('True Value', fontsize=textsize2)
        #ax.set_title('Confusion Matrix', fontsize=15, pad=15)
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"bilder/cn_matrix_dectree{title}.png", dpi=400, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()
    return rnd_clf

def rand_forest(X, y, plot=False, grid_search=False,show_plot=True,title =""):
    from sklearn.ensemble import RandomForestClassifier
    
    if grid_search:
        param_grid = {
        'n_estimators': [100, 200, 300,800,1200],
        'max_leaf_nodes': [5,10, 20, 30,80],
        'random_state': [42]
        }

        grid_search = GridSearchCV(RandomForestClassifier(n_jobs=-1), param_grid, cv=5)
        grid_search.fit(X, y)

        print("Beste Parameter:", grid_search.best_params_)
    
    rnd_clf = RandomForestClassifier(n_estimators=100, max_leaf_nodes=30, n_jobs=-1, random_state=42)  # this command (with its params) comes from the documentation
    rnd_clf.fit(X, y)
    
    if plot:
        feature_importance = rnd_clf.feature_importances_
        sorted_idx = feature_importance.argsort()  
        plt.barh(X.columns.values[sorted_idx], feature_importance[sorted_idx])
        plt.xlabel('feature importance')
        if show_plot:
            plt.show()
        else:
            plt.close()


        # Vorhersagen des Modells auf den Testdaten
        y_pred = rnd_clf.predict(X)

        # Berechnung der Confusion Matrix
        cm = confusion_matrix(y, y_pred,normalize=normalizeing)

        # Plot-Einstellungen
        plt.figure(figsize=(6, 6))
        sns.set(font_scale=textsize1)  # Größere Schrift für bessere Lesbarkeit
        ax = sns.heatmap(
            cm,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=['False', 'True'],
            yticklabels=['False', 'True'],
            cbar=False,
            linewidths=0.5,
            linecolor='gray',
            square=True
        )
        ax.set_xlabel('Predicted Value', fontsize=textsize2)
        ax.set_ylabel('True Value', fontsize=textsize2)
        #ax.set_title('Confusion Matrix', fontsize=15, pad=15)
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"bilder/cn_matrix_rndmforest{title}.png", dpi=400, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()
    return rnd_clf

def compare_algorithms(show_plot=True):
    
    alg = pd.read_csv("sortedcurves.csv")
    alg.rename(columns={"Unnamed: 0": "name"}, inplace=True)
    alg["variablealg"] = alg["True_Count"] >= 2
    alg = alg[["name", "variablealg"]]

    
    hand = pd.read_csv("Lichtkurven/galaxienotes.csv")
    hand["variablehand"] = hand["category"] >= 3
    hand = hand[["name", "variablehand"]]

    data = hand.merge(alg, on="name", how="inner")
    data.replace({"variablealg": {True: 1, False: 0}}, inplace=True)
    data.replace({"variablehand": {True: 1, False: 0}}, inplace=True)
    data["correct"] = data["variablealg"] == data["variablehand"]
    data.replace({"correct": {True: 1, False: 0}}, inplace=True)

    

    # Berechnung der Confusion Matrix
    cm = confusion_matrix(data["variablehand"].values, data["variablealg"].values,normalize=normalizeing)
    # Plot-Einstellungen
    plt.figure(figsize=(6, 6))
    sns.set(font_scale=textsize1)  # Größere Schrift für bessere Lesbarkeit
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=['False', 'True'],
        yticklabels=['False', 'True'],
        cbar=False,
        linewidths=0.5,
        linecolor='gray',
        square=True
    )
    ax.set_xlabel('Predicted Value', fontsize=textsize2)
    ax.set_ylabel('True Value', fontsize=textsize2)
    #ax.set_title('Confusion Matrix', fontsize=15, pad=15)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"bilder/cn_matrix_compare.png", dpi=400, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()
def contour_plot(X, y):


    from matplotlib.colors import ListedColormap
    cm_tree = ListedColormap(['#e58139', '#399de5'])  # colors are chosen to coincide with decision tree

    # Konvertiere X in ein NumPy-Array, falls es ein DataFrame ist
    if isinstance(X, pd.DataFrame):
        X = X.values

    model = rand_forest(X, y, plot=False,show_plot=False,grid_search=True)

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
    
    

def load_all():
    control = [True,False]
    alg = [knn, rand_forest, dec_tree]
    for a in tqdm(alg):
        console.print(f"[bold red]=============== Algorithm: {a.__name__} ===============\n[/bold red]")
        for c in control:
            X_train,X_test,y_train,y_test = load_data(c)
            if c:
                title = "alg"
            else:
                title = "hand"
            model = a(X_train, y_train, plot=True,show_plot=False,grid_search=True ,title = title)
            print(f"===== {title} =====")
            if c:
                print(f"\nTrain Score: {model.score(X_train, y_train)}\nTest Score: {model.score(X_test, y_test)}\n")
            else:
                print(f"\nTrain Score: {model.score(X_train, y_train)}\nTest Score: {model.score(X_test, y_test)}\n")
                       
    compare_algorithms(show_plot=False)
    




load_all()

#X_train,X_test,y_train,y_test = load_data(False)
#model = rand_forest(X_train, y_train, plot = True,show_plot=True,grid_search=True)

#contour_plot(X_train[["amplitude","periodicFast"]], y_train)








