import pandas as pd
import numpy as np
from collections import OrderedDict

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

def get_ohe_data(df, S, Y, S_under, Y_desire):
    df_int = df.select_dtypes(['float', 'integer']).values
    continuous_columns_list = list(df.select_dtypes(['float', 'integer']).columns)
    ##############################################################
    scaler = QuantileTransformer(n_quantiles=2000, output_distribution='uniform')
    df_int = scaler.fit_transform(df_int)

    df_cat = df.select_dtypes('object')
    df_cat_names = list(df.select_dtypes('object').columns)
    numerical_array = df_int
    ohe = OneHotEncoder()
    ohe_array = ohe.fit_transform(df_cat)

    cat_lens = [i.shape[0] for i in ohe.categories_]
    discrete_columns_ordereddict = OrderedDict(zip(df_cat_names, cat_lens))

    S_start_index = len(continuous_columns_list) + sum(
        list(discrete_columns_ordereddict.values())[:list(discrete_columns_ordereddict.keys()).index(S)])
    Y_start_index = len(continuous_columns_list) + sum(
        list(discrete_columns_ordereddict.values())[:list(discrete_columns_ordereddict.keys()).index(Y)])

    if ohe.categories_[list(discrete_columns_ordereddict.keys()).index(S)][0] == S_under:
        underpriv_index = 0
        priv_index = 1
    else:
        underpriv_index = 1
        priv_index = 0
    if ohe.categories_[list(discrete_columns_ordereddict.keys()).index(Y)][0] == Y_desire:
        desire_index = 0
        undesire_index = 1
    else:
        desire_index = 1
        undesire_index = 0

    final_array = np.hstack((numerical_array, ohe_array.toarray()))
    return ohe, scaler, discrete_columns_ordereddict, continuous_columns_list, final_array, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index