import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, RepeatedStratifiedKFold, StratifiedKFold 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import itertools

# read preprocessed data
data_mous= pd.read_csv("data_processed_moussieri.csv")
data_och= pd.read_csv("data_processed_ochruros.csv")
data_pho= pd.read_csv("data_processed_phoenicurus.csv")

# extract the enviremental data
X_mous= data_mous.drop(["Longitude", "Latitude", "presence"], axis=1)
X_och= data_och.drop(["Longitude", "Latitude", "presence"], axis=1)
X_pho= data_pho.drop(["Longitude", "Latitude", "presence"], axis=1)

# extract the target variable
Y_mous= data_mous["presence"]
Y_och= data_och["presence"]
Y_pho= data_pho["presence"]


## defining the models
classifiers_objects= [
    KNeighborsClassifier(),
    SVC(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    DecisionTreeClassifier(),
    QuadraticDiscriminantAnalysis(),
    RandomForestClassifier(),
    MLPClassifier()
]
classifiers_names= ["KNN", "SVM", "AB", "GB", "DT", 'QDA', "RF", "MLP"]


# This function return us the 'correct wrong matrix' that we saw in paper of diversity
def correct_wrong_matrix(atual_y, cls1_y, cls2_y, proba=False):

    mx= np.zeros((2,2))
    for i, actaul_class in enumerate(atual_y):
        if (cls1_y[i] == actaul_class and cls2_y[i] == actaul_class):
            mx[0,0] +=1
        elif (cls1_y[i] == actaul_class and cls2_y[i] != actaul_class):
            mx[0,1] +=1
        elif (cls1_y[i] != actaul_class and cls2_y[i] == actaul_class):
            mx[1,0] +=1
        elif (cls1_y[i] != actaul_class and cls2_y[i] != actaul_class):
            mx[1,1] +=1
    if (proba == True):
        N= mx.sum()
        mx[0,0] /= N
        mx[1,0] /= N
        mx[0,1] /= N
        mx[1,1] /= N
    return mx


# this function takes 'correct wrong matrix' as parameter and returns us Q statistic value
def calculate_Q_statistic(mx):
    ad= mx[0,0]*mx[1,1]
    bc= mx[1,0]*mx[0,1]
    return ( (ad - bc)/(ad + bc) )


# this function calculate the average Q statictic for set of classifiers
def calculate_combin_Q(actual_y, y_pred_list):
    # y_pred_list is list of vectors (classifiers results)
    #====== from y_pred_list we make all possible combinations of lenght 2
    y_2combins=[]
    for comb in itertools.combinations(y_pred_list, 2):
        y_2combins.append(comb)
    #======

    #====== we iterate all the combination of 2, calculate thier Q statistic
    Q_lists=[]
    for comb_elem in y_2combins:
        mx= correct_wrong_matrix(actual_y, comb_elem[0], comb_elem[1], proba=True)
        Q= calculate_Q_statistic(mx)
        Q_lists.append(Q)
    #======

    #====== we return the average Q statistic
    return ( sum(Q_lists) / len(Q_lists))

# final results (table that summarizes all)
def diversity_cal(actual_y, y_pred_list, classifiers_names, comb_size):
    #===== 
    # y_pred_list must have results of 7 models in the same order of models_names
    y_pred_combins=[]
    # models_names_combins=[]
    y_2combins_names=[]
    y_2combins=[]
    for comb in itertools.combinations(y_pred_list, comb_size):
        y_2combins.append(comb)
    for comb in itertools.combinations(classifiers_names, comb_size):
        y_2combins_names.append(comb)
    #=====

    # for each ensemble we alculate average Q and accuracy
    data_frame = pd.DataFrame(columns=["ens","Mean diversity(Q)"])
    for i, y_pred_combins_elem in enumerate(y_2combins):
        Q_5folds=[]
        for j in range(10):
            # using precident function we calculate ensemble Q
            y_pred_list__=[]
            for i_cb in range(len(y_pred_combins_elem)):
                y_pred_list__.append(y_pred_combins_elem[i_cb][j])
            Q= calculate_combin_Q(actual_y[j], y_pred_list__)
            Q_5folds.append(Q)
      
        avg_Q= np.array(Q_5folds).mean()
    
        # we add new row to dataframe
        data_frame.loc[len(data_frame.index)] = [str(y_2combins_names[i]), round(avg_Q, 3)] 
    return data_frame


## fiting the models to calculate diversity based on the results obtained
scores_mous= []
scores_och= []
scores_pho= []
skf = StratifiedKFold(n_splits=10, random_state=111, shuffle=True)
for i, model in enumerate(classifiers_objects):
    scores_1 = cross_val_predict(model, X_mous, Y_mous, cv=skf,  n_jobs=-1)
    scores_2 = cross_val_predict(model, X_och, Y_och, cv=skf,  n_jobs=-1)
    scores_3 = cross_val_predict(model, X_pho, Y_pho, cv=skf,  n_jobs=-1)
    
    # geting predicted scores
    scores_mous.append([scores_1[j] for i, j in skf.split(X_mous,Y_mous)])
    scores_och.append([scores_2[j] for i, j in skf.split(X_och,Y_och)])
    scores_pho.append([scores_3[j] for i, j in skf.split(X_pho,Y_pho)])

# geting acual scores
score_act_mous = [np.array(Y_mous)[j] for i, j in skf.split(X_mous,Y_mous)]
score_act_och = [np.array(Y_och)[j] for i, j in skf.split(X_och,Y_och)]
score_act_pho = [np.array(Y_pho)[j] for i, j in skf.split(X_pho,Y_pho)]


# this calculate the diversity of all models pairs
# takes in parameters: acual scores, predicted scores, classifiers names, and combination size
df_2= diversity_cal(score_act, scores_pred, classifiers_names, 2)
df= df.sort_values(by=['Mean diversity(Q)'], ascending=True)

