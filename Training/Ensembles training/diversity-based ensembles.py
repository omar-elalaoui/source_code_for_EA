import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


# read the preprocessed data
data_mous= pd.read_csv("data_processed_moussieri.csv")
data_och= pd.read_csv("data_processed_ochruros.csv")
data_pho= pd.read_csv("data_processed_phoenicurus.csv")

# extract the enviremental data to train the models 
X_mous= data_mous.drop(["Longitude", "Latitude", "presence"], axis=1)
X_och= data_och.drop(["Longitude", "Latitude", "presence"], axis=1)
X_pho= data_pho.drop(["Longitude", "Latitude", "presence"], axis=1)

# ectract the target variable
Y_mous= data_mous["presence"]
Y_och= data_och["presence"]
Y_pho= data_pho["presence"]


## defining the models
base_learners_mous= [
    ("DT", DecisionTreeClassifier()),
    ("QDA", QuadraticDiscriminantAnalysis()),
    ("AB", AdaBoostClassifier()),
    ("KNN", KNeighborsClassifier()),
    ("SVM", SVC(probability=True)),
    ("MLP", MLPClassifier()),
    ("GB", GradientBoostingClassifier()),
    ("RF", RandomForestClassifier())
    
]
base_learners_och= [
    ("DT", DecisionTreeClassifier()),
    ("QDA", QuadraticDiscriminantAnalysis()),
    ("SVM", SVC(probability=True)),
    ("AB", AdaBoostClassifier()),
    ("KNN", KNeighborsClassifier()),
    ("MLP", MLPClassifier()),
    ("RF", RandomForestClassifier()),
    ("GB", GradientBoostingClassifier())
]

base_learners_pho= [
    ("AB", AdaBoostClassifier()),
    ("QDL", QuadraticDiscriminantAnalysis()),
    ("DT", DecisionTreeClassifier()),
    ("MLP", MLPClassifier()),
    ("KNN", KNeighborsClassifier()),
    ("SVM", SVC(probability=True)),
    ("GB", GradientBoostingClassifier()),
    ("RF", RandomForestClassifier())
]


soft_voting_objects_mous= []
soft_voting_objects_och= []
soft_voting_objects_pho= []
for i in range(7):
    i=i+1
    soft_voting_objects_mous.append(VotingClassifier(estimators=base_learners_mous[:i+1], voting='soft'))
    soft_voting_objects_och.append(VotingClassifier(estimators=base_learners_och[:i+1], voting='soft'))
    soft_voting_objects_pho.append(VotingClassifier(estimators=base_learners_pho[:i+1], voting='soft'))


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


## fiting the ensembles on 10 folds CV
scores_mous= []
scores_och= []
scores_pho= []
for i, model in enumerate(soft_voting_objects_mous):
    scores_1 = cross_validate(solft_voting_objects_mous[i], X_mous, Y_mous, cv=10, scoring=scores_cv,  n_jobs=-1, error_score="raise")
    scores_2 = cross_validate(solft_voting_objects_och[i], X_och, Y_och, cv=10, scoring=scores_cv,  n_jobs=-1, error_score="raise")
    scores_3 = cross_validate(solft_voting_objects_pho[i], X_pho, Y_pho, cv=10, scoring=scores_cv,  n_jobs=-1, error_score="raise")
    scores_mous.append(scores_1)
    scores_och.append(scores_2)
    scores_pho.append(scores_3)


# function to display results in dataframe
classifiers_names= ['E2-SBD', 'E3-SBD', 'E4-SBD', 'E5-SBD', 'E6-SBD', 'E7-SBD', 'E8-SBD']
def results_table(classifiers_scores_, classifiers_names_):
    data_frame = pd.DataFrame(columns=['Accuracy', 'Sensitivity', 'Specificity', 'AUC', 'Kappa', 'Tss', 'acc std'])
    for i, model_scores in enumerate(classifiers_scores_):
        data_frame.loc[classifiers_names_[i]] = ["", "", "", "", "", "",""]    
        data_frame['Accuracy'][classifiers_names_[i]]= round(model_scores['test_accuracy'].mean()*100, 1)
        data_frame['Sensitivity'][classifiers_names_[i]]= round(model_scores['test_sensitivity'].mean()*100, 1)
        data_frame['Specificity'][classifiers_names_[i]]= round(model_scores['test_specificity'].mean()*100, 1)
        data_frame['AUC'][classifiers_names_[i]]= round(model_scores['test_auc'].mean()*100, 1)
        data_frame['Kappa'][classifiers_names_[i]]= round(model_scores['test_kappa'].mean(), 2)
        data_frame['Tss'][classifiers_names_[i]]= round(np.median(model_scores['test_tss']), 3)
    return data_frame



df1= results_table(scores_mous, classifiers_names)
df2= results_table(scores_och, classifiers_names)
df3= results_table(scores_pho, classifiers_names)



# function to save results dataframe in excel files
def save_df_as_excel(path, dfs, names, startC=0, startR=0, index=True):
    writer = pd.ExcelWriter(path)
    for i, df in enumerate(dfs):
        df.to_excel(writer, names[i], startcol=startC, startrow=startR, index= index)
    writer.save()
    writer.close()
    
# function to save results for SK test
def generate_excel_for_SK(scores, models_names):
    data_frame = pd.DataFrame(columns=['KFOLD', 'AUC', 'MODEL'],)
    for i, score in enumerate(scores):
        for j, score_acc in enumerate(score["test_auc"]):
            new_row = {'KFOLD': j+1, 'AUC': round(score_acc*100, 1), 'MODEL':models_names[i],}
            data_frame = data_frame.append(new_row, ignore_index=True)
    return data_frame

#function to save results for generating 95% CI
def generate_excel_for_CI(scores_4, species_names, models_names):
    data_frame = pd.DataFrame(columns=['Species', 'AUC', 'Models'],)
    for s, scores in enumerate(scores_4): 
        for i, score in enumerate(scores):
            for j, score_acc in enumerate(score["test_auc"]):
                new_row = {'Species': species_names[s], 'AUC': round(score_acc*100, 1), 'Models':models_names[i]}
                data_frame = data_frame.append(new_row, ignore_index=True)
    return data_frame