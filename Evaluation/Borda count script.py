import numpy as np
import pandas as pd


# implementation of Borca count voting method
def borda_count(df_s):  
  df_= df_s.copy()
  df_columns= df_.columns.values.tolist()
  df_columns_len= len(df_.columns)-1
  df_rows_len= df_.shape[0]
  for i in range(df_columns_len):
    df_[df_columns[i+1]]= df_[df_columns[i+1]].rank(ascending=False).astype(int)
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

def save_df_as_excel(path, dfs, names, startC=0, startR=0, index=True):
    writer = pd.ExcelWriter(path)
    for i, df in enumerate(dfs):
        df.to_excel(writer, names[i], startcol=startC, startrow=startR, index= index)
    writer.save()
    writer.close()


# read results tables 
results_mous= pd.read_excel("results_table.xlsx", sheet_name="mous")
results_och= pd.read_excel("results_table.xlsx", sheet_name="och")
results_pho= pd.read_excel("results_table.xlsx", sheet_name="pho")

# generate borda count ranks
df1= borda_count(results_mous)
df2= borda_count(results_och)
df3= borda_count(results_pho)