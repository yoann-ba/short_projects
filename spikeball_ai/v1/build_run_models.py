


#%% imports


import numpy as np
# import matplotlib.pyplot as plt

# import os
from pathlib import Path

import time


# -- scikit learn
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# -- own scripts
import general_utils as utils


#%% funcs


def compute_scores(y_pred, y_true, beta = 1):
    
    accuracy = accuracy_score(y_true, y_pred)
    precision_1 = precision_score(y_true, y_pred, zero_division=1)
    precision_0 = precision_score(y_true, y_pred, zero_division=1, pos_label = 0)
    recall_1 = recall_score(y_true, y_pred)
    recall_0 = recall_score(y_true, y_pred, pos_label = 0)
    # 
    f_beta_1 = fbeta_score(y_true, y_pred, zero_division=1, beta=beta)
    f_beta_0 = fbeta_score(y_true, y_pred, zero_division=1, beta=beta, pos_label = 0)
    # 
    clean_print = f"{accuracy*100:3.2f}% |"
    clean_print += f"{precision_1*100:3.2f}%/{recall_1*100:3.2f}% |"
    clean_print += f"{precision_0*100:3.2f}%/{recall_0*100:3.2f}% |"
    clean_print += f"{f_beta_1*100:3.2f}%, {f_beta_0*100:3.2f}% "
    score_dict = {
        "acc": accuracy, 
        "prec_1": precision_1, 
        "prec_0": precision_0, 
        "recall_1": recall_1, 
        "recall_0": recall_0, 
        "f_beta_1": f_beta_1, 
        "f_beta_0": f_beta_0, 
        "clean_print": clean_print, 
        }
    
    return score_dict



#%% 

names = [
    "Nearest N",
    # "Linear SVM", # slow
    # "RBF SVM", #too slow
    # Gaussian Process # wanted to do a X * X computation so nope
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    # "QDA", # didn't like float16
    "Log Reg clf", 
    ]
classifiers = [
    KNeighborsClassifier(5),
    # SVC(kernel="linear", C=0.025),
    # SVC(gamma=2, C=1),
    RandomForestClassifier(),
    MLPClassifier(hidden_layer_sizes = (10, 30, 10), max_iter=1000),
    AdaBoostClassifier(algorithm="SAMME"),
    GaussianNB(),
    # QuadraticDiscriminantAnalysis(),
    LogisticRegression(max_iter = 1000), 
    ]


#TODO clean hard
def run_all_models(datasets_path, names=names, classifiers=classifiers):
    # 
    fitted_models = {}
    print("Acc. | Prec 1/Rec 1  | Prec 0/Rec 0 | F1 1, F1 0")
    for data_mode in ['5d', 'rgb']:
        for label in utils.colors_to_labels:
            print(f" - - - {label} {data_mode} - - - ")
            temp_path = Path(datasets_path + f"{label}_{data_mode}_full.npy")
            X = np.load(temp_path)
            temp_path = Path(datasets_path + f"{label}_y_{data_mode}_full.npy")
            Y = np.load(temp_path)
            # 
            rskf = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 1)
            skip_skf = 0
            for i_skf, (train_index, test_index) in enumerate(rskf.split(X, Y)):
                # print(f"  > split {i_skf}")
                x_train, x_test = X[train_index], X[test_index]
                y_train, y_test = Y[train_index], Y[test_index]
                # 
                for i_clf in range(len(classifiers)):
                    model = clone(classifiers[i_clf])
                    model_name = names[i_clf]
                    print(f"{model_name:20s}", end = "")
                    # 
                    time_a = time.time()
                    model.fit(x_train, y_train)
                    preds = model.predict(x_test)
                    score_dict = compute_scores(preds, y_test)
                    time_b = time.time()
                    print(score_dict['clean_print'] + f"|| {time_b - time_a:.2f}s")
                    # 
                    #TODO this is a horror 
                    fitted_models[f"{label}_{data_mode}"] = model
                # 
                #TODO to remove, just to skip skf
                skip_skf += 1
                if skip_skf > 1:
                    continue
            # 
        # 
    
    return fitted_models


#%% sandbox

# datasets_path = "E:\Python_Data\spikeball\spikeball_v1\shaped_datasets/3_label_unique_4reduc/"


#%%

# default, run all
# run_all_models(datasets_path = datasets_path)

# test = run_all_models(datasets_path = datasets_path, 
#                       names = ['RFC'], 
#                       classifiers = [RandomForestClassifier(class_weight = 'balanced')])




























































































