import pandas as pd
import numpy as np
from get_ohe_data import get_ohe_data

lista_colunas = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week','race','sex','income']

df1 = pd.read_csv('src/adult.csv')
df2 = pd.read_csv('tests/mdr/fake_data_wgan_mdr_0_0.csv')
#df2 = pd.read_csv('tests/mdr/fake_data_3TS_0.5_0.csv')
df2.drop('predict', axis = 1, inplace = True)
df1 = df1[lista_colunas].head(30000)
df2 = df2[lista_colunas].head(30000)

S = 'sex'
Y = 'income'
S_under = ' Female'
Y_desire = ' >50K'

ohe_1, scaler_1, discrete_columns_ordereddict_1, continuous_columns_list_1, final_array_1, S_start_index_1, Y_start_index_1, underpriv_index_1, priv_index_1, undesire_index_1, desire_index_1 = get_ohe_data(df1, S, Y, S_under, Y_desire)
ohe_2, scaler_2, discrete_columns_ordereddict_2, continuous_columns_list_2, final_array_2, S_start_index_2, Y_start_index_2, underpriv_index_2, priv_index_2, undesire_index_2, desire_index_2 = get_ohe_data(df2, S, Y, S_under, Y_desire)

print(final_array_1.shape,final_array_2.shape)

from scipy.special import rel_entr
from scipy.stats import wasserstein_distance

wdist_feature = []
for i in range(15):
    wdist = wasserstein_distance(final_array_1[i],final_array_2[i])
    wdist_feature.append(wdist)

print(np.array(wdist_feature).sum())
