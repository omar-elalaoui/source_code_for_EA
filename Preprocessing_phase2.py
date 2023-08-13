import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest, SelectPercentile

# read the data
data = pd.read_excel('input data/data.xlsx')

#data shape & and outcome class
print("data dimensions: ",data.shape)
print("--------")
print("Presence class:")
print("0 ==>", len(data[data["presence"] == 0]))
print("1 ==>", len(data[data["presence"] == 1]))
print("--------")
print("data types:\n\n", data.dtypes)


# check for missing values
ms_values= data.isnull().sum()
## drop missing values
data= data.dropna()


## function to identify bounaries (for outliers detection)
def find_skewed_boundaries(df, variable, distance):
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)
    return upper_boundary, lower_boundary


predictors_variables = data.drop(["Longitude", "Latitude", "presence"], axis=1)
target_var= data["presence"]
# make a copy of the main dataframe
data_cp1= data.copy()

## detect outliers
total_outliers=0
for v in vars:
    up, low = find_skewed_boundaries(data, v, 3)
    outliers = data.loc[np.where(data[v] > up, True, np.where(data[v] < low, True, False))]
    total_outliers += len(outliers)
    print(v, " :", len(outliers))
    data_p0.loc[data_p0[v] < low, v] = np.nan
    data_p0.loc[data_p0[v] > up, v] = np.nan
print("--------")
print("total outlier: {} ==> {}%".format(total_outliers, round((total_outliers/len(data)*100), 2) ))
data_cp2= data_cp1.dropna()


# dictionary that holds variable as key and MIN_score as value
scores = {}
for i, j in zip(feature_names, MI_score):
    scores[i] = j

# list that will contain tuples (correlated pairs of variables)
correlated_pairs = []

# identify correlated pairs (measuring the pearson correlation coefficient for each pairs)
corr_matrix = data_num.corr()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if (abs(corr_matrix.iloc[i, j]) > 0.8):
            correlated_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))


#When two variables are highly correlated, the one with the lower MI score was removed
del_col = []
for pair in correlated_pairs:
    if scores[pair[0]] < scores[pair[1]]:
        del_col.append(pair[0])
    else:
        del_col.append(pair[1])
del_col = set(del_col)
data_cp3 = data_cp2.drop(del_col, axis=1)


## Z-score normalization
num_columns= data_cp3.drop(["Longitude", "Latitude", "presence", "LC_ext"], axis=1).columns
data_cp4= data_cp3.copy()
for col in num_columns:
    data_cp4[col] = (data_cp4[col] - data_cp4[col].mean())/data_cp4[col].std(ddof=0)