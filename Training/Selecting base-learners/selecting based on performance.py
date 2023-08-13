import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, RepeatedStratifiedKFold, StratifiedKFold 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import itertools

# function to calculate borda count of the eight models
def borda_count(df_s):  
  df_= df_s.copy()
  df_columns= df_.columns.values.tolist()
  df_columns_len= len(df_.columns)
  df_rows_len= df_.shape[0]
  for i in range(df_columns_len):
    df_[df_columns[i]]= df_[df_columns[i]].rank(ascending=False).astype(int)
  temp_lists= df_.iloc[:, 1:].values.tolist()
  temp_list_2=[]
  temp_scores=[]
  for i, list_ in enumerate(temp_lists):
    listo_= [0]*df_rows_len
    score=0
    for j in range(df_rows_len):
      listo_[j]=0
      for elem in list_:
        if elem == j+1:
          listo_[j] += 1
          score += (df_rows_len-j)
    temp_list_2.append(str(listo_))
    temp_scores.append(score)
  df_["Position"]=temp_list_2
  df_["Score"]= temp_scores
  df_["Rank"]= df_["Score"].rank(ascending=False, method='min').astype(int)
  temp=df_.sort_values(by=['Score'], ascending=False)
  return temp



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



# defining the metrics
from sklearn.metrics import make_scorer, recall_score, accuracy_score, roc_auc_score, cohen_kappa_score
def tss_scorer(y_actual, y_pred):
    TSS= recall_score(y_actual, y_pred, pos_label=0) + recall_score(y_actual, y_pred, pos_label=1) - 1
    return TSS
scores_cv= {
    'accuracy': make_scorer(accuracy_score),
    'sensitivity': make_scorer(recall_score, pos_label=1),
    'specificity': make_scorer(recall_score, pos_label=0),
    'auc': make_scorer(roc_auc_score),
    'kappa': make_scorer(cohen_kappa_score),
    'tss': make_scorer(tss_scorer, greater_is_better=True)
}


## fiting the models
scores_mous= []
scores_och= []
scores_pho= []
for i, model in enumerate(classifiers_objects):
    scores_1 = cross_validate(model, X_mous, Y_mous, cv=10, scoring=scores_cv,  n_jobs=-1)
    scores_2 = cross_validate(model, X_och, Y_och, cv=10, scoring=scores_cv,  n_jobs=-1)
    scores_3 = cross_validate(model, X_pho, Y_pho, cv=10, scoring=scores_cv,  n_jobs=-1)
    
    scores_mous.append(scores_1)
    scores_och.append(scores_2)
    scores_pho.append(scores_3)


# function to convert results in dataframe
def results_table(classifiers_scores_, classifiers_names_):
    data_frame = pd.DataFrame(columns=['Accuracy', 'Sensitivity', 'Specificity', 'AUC', 'Kappa', 'Tss'])
    for i, model_scores in enumerate(classifiers_scores_):
        data_frame.loc[classifiers_names_[i]] = ["", "", "", "", "", ""]    
        data_frame['Accuracy'][classifiers_names_[i]]= round(model_scores['test_accuracy'].mean()*100, 1)
        data_frame['Sensitivity'][classifiers_names_[i]]= round(model_scores['test_sensitivity'].mean()*100, 1)
        data_frame['Specificity'][classifiers_names_[i]]= round(model_scores['test_specificity'].mean()*100, 1)
        data_frame['AUC'][classifiers_names_[i]]= round(model_scores['test_auc'].mean()*100, 1)
        data_frame['Kappa'][classifiers_names_[i]]= round(model_scores['test_kappa'].mean(), 2)
        data_frame['Tss'][classifiers_names_[i]]= round(np.median(model_scores['test_tss']), 3)
    return data_frame

# represent results in dataframe
df1= results_table(scores_mous, classifiers_names)
df2= results_table(scores_och, classifiers_names)
df3= results_table(scores_pho, classifiers_names)

# rank the models with borda count based on the six metrics
df1_bc= borda_count(df1)
df2_bc= borda_count(df2)
df3_bc= borda_count(df3)