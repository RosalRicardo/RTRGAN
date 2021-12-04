import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

df_nu00 = pd.read_csv('tests/mdr/fake_data_wgan_mdr_0_0.csv')

df_nu00['TPR_FEMALE'] = (df_nu00['income'] == ' >50K') & (df_nu00['sex'] == ' Female') & (df_nu00['predict'] == 1)
df_nu00['TPR_MALE'] = (df_nu00['income'] == ' >50K') & (df_nu00['sex'] == ' Male') & (df_nu00['predict'] == 1)

df_TPR_nu00 = df_nu00[['TPR_MALE','TPR_FEMALE']]
df_TPR_nu00 = df_TPR_nu00.to_numpy()
print(np.sum(df_TPR_nu00,axis = 0))

df_nu01 = pd.read_csv('tests/mdr/fake_data_wgan_mdr_0.1_0.csv')

df_nu01['TPR_FEMALE'] = (df_nu01['income'] == ' >50K') & (df_nu01['sex'] == ' Female') & (df_nu01['predict'] == 1)
df_nu01['TPR_MALE'] = (df_nu01['income'] == ' >50K') & (df_nu01['sex'] == ' Male') & (df_nu01['predict'] == 1)

df_TPR_nu01 = df_nu01[['TPR_MALE','TPR_FEMALE']]
df_TPR_nu01 = df_TPR_nu01.to_numpy()
print(np.sum(df_TPR_nu01,axis = 0))

df_nu03 = pd.read_csv('tests/mdr/fake_data_wgan_mdr_0.3_0.csv')

df_nu03['TPR_FEMALE'] = (df_nu03['income'] == ' >50K') & (df_nu03['sex'] == ' Female') & (df_nu03['predict'] == 1)
df_nu03['TPR_MALE'] = (df_nu03['income'] == ' >50K') & (df_nu03['sex'] == ' Male') & (df_nu03['predict'] == 1)

df_TPR_nu03 = df_nu03[['TPR_MALE','TPR_FEMALE']]
df_TPR_nu03 = df_TPR_nu03.to_numpy()
print(np.sum(df_TPR_nu03,axis = 0))

df_nu05 = pd.read_csv('tests/mdr/fake_data_wgan_mdr_0.5_0.csv')

df_nu05['TPR_FEMALE'] = (df_nu05['income'] == ' >50K') & (df_nu05['sex'] == ' Female') & (df_nu05['predict'] == 1)
df_nu05['TPR_MALE'] = (df_nu05['income'] == ' >50K') & (df_nu05['sex'] == ' Male') & (df_nu05['predict'] == 1)

df_TPR_nu05 = df_nu05[['TPR_MALE','TPR_FEMALE']]
df_TPR_nu05 = df_TPR_nu05.to_numpy()
print(np.sum(df_TPR_nu05,axis = 0))

