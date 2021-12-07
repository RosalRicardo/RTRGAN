import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

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

print('================ Depois das 3 fases de treino ============================')

print('Testes com três etapas de treino e nu igual 0')
df_nu00_3TS = pd.read_csv('tests/mdr/fake_data_wgan_mdr_0_0.csv')

df_nu00_3TS['TPR_FEMALE_VP'] = (df_nu00_3TS['income'] == ' >50K') & (df_nu00_3TS['sex'] == ' Female') & (df_nu00_3TS['predict'] == 1)
df_nu00_3TS['TPR_MALE_VP'] = (df_nu00_3TS['income'] == ' >50K') & (df_nu00_3TS['sex'] == ' Male') & (df_nu00_3TS['predict'] == 1)
df_nu00_3TS['TPR_FEMALE_FP'] = (df_nu00_3TS['income'] != ' >50K') & (df_nu00_3TS['sex'] == ' Female') & (df_nu00_3TS['predict'] == 0)
df_nu00_3TS['TPR_MALE_FP'] = (df_nu00_3TS['income'] != ' >50K') & (df_nu00_3TS['sex'] == ' Male') & (df_nu00_3TS['predict'] == 0)
df_nu00_3TS = df_nu00_3TS[['TPR_FEMALE_VP','TPR_MALE_VP','TPR_FEMALE_FP','TPR_MALE_FP']]
df_nu00_3TS = df_nu00_3TS.to_numpy()
print(np.sum(df_nu00_3TS,axis = 0),np.sum(df_nu00_3TS))

print('Testes com três etapas de treino e nu igual 0.1')
df_nu01_3TS = pd.read_csv('tests/mdr/fake_data_3TS_0.1_0.csv')

df_nu01_3TS['TPR_FEMALE_VP'] = (df_nu01_3TS['income'] == ' >50K') & (df_nu01_3TS['sex'] == ' Female') & (df_nu01_3TS['predict'] == 1)
df_nu01_3TS['TPR_MALE_VP'] = (df_nu01_3TS['income'] == ' >50K') & (df_nu01_3TS['sex'] == ' Male') & (df_nu01_3TS['predict'] == 1)
df_nu01_3TS['TPR_FEMALE_FP'] = (df_nu01_3TS['income'] != ' >50K') & (df_nu01_3TS['sex'] == ' Female') & (df_nu01_3TS['predict'] == 0)
df_nu01_3TS['TPR_MALE_FP'] = (df_nu01_3TS['income'] != ' >50K') & (df_nu01_3TS['sex'] == ' Male') & (df_nu01_3TS['predict'] == 0)
df_nu01_3TS = df_nu01_3TS[['TPR_FEMALE_VP','TPR_MALE_VP','TPR_FEMALE_FP','TPR_MALE_FP']]
df_nu01_3TS = df_nu01_3TS.to_numpy()
print(np.sum(df_nu01_3TS,axis = 0),np.sum(df_nu01_3TS))

print('Testes com três etapas de treino e nu igual 0.3')
df_nu03_3TS = pd.read_csv('tests/mdr/fake_data_3TS_0.3_0.csv')

df_nu03_3TS['TPR_FEMALE_VP'] = (df_nu03_3TS['income'] == ' >50K') & (df_nu03_3TS['sex'] == ' Female') & (df_nu03_3TS['predict'] == 1)
df_nu03_3TS['TPR_MALE_VP'] = (df_nu03_3TS['income'] == ' >50K') & (df_nu03_3TS['sex'] == ' Male') & (df_nu03_3TS['predict'] == 1)
df_nu03_3TS['TPR_FEMALE_FP'] = (df_nu03_3TS['income'] != ' >50K') & (df_nu03_3TS['sex'] == ' Female') & (df_nu03_3TS['predict'] == 0)
df_nu03_3TS['TPR_MALE_FP'] = (df_nu03_3TS['income'] != ' >50K') & (df_nu03_3TS['sex'] == ' Male') & (df_nu03_3TS['predict'] == 0)
df_nu03_3TS = df_nu03_3TS[['TPR_FEMALE_VP','TPR_MALE_VP','TPR_FEMALE_FP','TPR_MALE_FP']]
df_nu03_3TS = df_nu03_3TS.to_numpy()
print(np.sum(df_nu03_3TS,axis = 0),np.sum(df_nu03_3TS))

print('Testes com três etapas de treino e nu igual 0.5')
df_nu05_3TS = pd.read_csv('tests/mdr/fake_data_3TS_0.5_0.csv')

df_nu05_3TS['TPR_FEMALE_VP'] = (df_nu05_3TS['income'] == ' >50K') & (df_nu05_3TS['sex'] == ' Female') & (df_nu05_3TS['predict'] == 1)
df_nu05_3TS['TPR_MALE_VP'] = (df_nu05_3TS['income'] == ' >50K') & (df_nu05_3TS['sex'] == ' Male') & (df_nu05_3TS['predict'] == 1)
df_nu05_3TS['TPR_FEMALE_FP'] = (df_nu05_3TS['income'] != ' >50K') & (df_nu05_3TS['sex'] == ' Female') & (df_nu05_3TS['predict'] == 0)
df_nu05_3TS['TPR_MALE_FP'] = (df_nu05_3TS['income'] != ' >50K') & (df_nu05_3TS['sex'] == ' Male') & (df_nu05_3TS['predict'] == 0)
df_nu05_3TS = df_nu05_3TS[['TPR_FEMALE_VP','TPR_MALE_VP','TPR_FEMALE_FP','TPR_MALE_FP']]
df_nu05_3TS = df_nu05_3TS.to_numpy()
print(np.sum(df_nu05_3TS,axis = 0),np.sum(df_nu05_3TS))

print('Testes com três etapas de treino e nu igual 0.7')
df_nu07_3TS = pd.read_csv('tests/mdr/fake_data_3TS_0.7_0.csv')

df_nu07_3TS['TPR_FEMALE_VP'] = (df_nu07_3TS['income'] == ' >50K') & (df_nu07_3TS['sex'] == ' Female') & (df_nu07_3TS['predict'] == 1)
df_nu07_3TS['TPR_MALE_VP'] = (df_nu07_3TS['income'] == ' >50K') & (df_nu07_3TS['sex'] == ' Male') & (df_nu07_3TS['predict'] == 1)
df_nu07_3TS['TPR_FEMALE_FP'] = (df_nu07_3TS['income'] != ' >50K') & (df_nu07_3TS['sex'] == ' Female') & (df_nu07_3TS['predict'] == 0)
df_nu07_3TS['TPR_MALE_FP'] = (df_nu07_3TS['income'] != ' >50K') & (df_nu07_3TS['sex'] == ' Male') & (df_nu07_3TS['predict'] == 0)
df_nu07_3TS = df_nu07_3TS[['TPR_FEMALE_VP','TPR_MALE_VP','TPR_FEMALE_FP','TPR_MALE_FP']]
df_nu07_3TS = df_nu07_3TS.to_numpy()
print(np.sum(df_nu07_3TS,axis = 0),np.sum(df_nu07_3TS))

print('Testes com três etapas de treino e nu igual 0.9')
df_nu09_3TS = pd.read_csv('tests/mdr/fake_data_3TS_0.9_0.csv')

df_nu09_3TS['TPR_FEMALE_VP'] = (df_nu09_3TS['income'] == ' >50K') & (df_nu09_3TS['sex'] == ' Female') & (df_nu09_3TS['predict'] == 1)
df_nu09_3TS['TPR_MALE_VP'] = (df_nu09_3TS['income'] == ' >50K') & (df_nu09_3TS['sex'] == ' Male') & (df_nu09_3TS['predict'] == 1)
df_nu09_3TS['TPR_FEMALE_FP'] = (df_nu09_3TS['income'] != ' >50K') & (df_nu09_3TS['sex'] == ' Female') & (df_nu09_3TS['predict'] == 0)
df_nu09_3TS['TPR_MALE_FP'] = (df_nu09_3TS['income'] != ' >50K') & (df_nu09_3TS['sex'] == ' Male') & (df_nu09_3TS['predict'] == 0)
df_nu09_3TS = df_nu09_3TS[['TPR_FEMALE_VP','TPR_MALE_VP','TPR_FEMALE_FP','TPR_MALE_FP']]
df_nu09_3TS = df_nu09_3TS.to_numpy()
print(np.sum(df_nu09_3TS,axis = 0),np.sum(df_nu09_3TS))

print('matriz de confusão dos dados reais')
adult_df = pd.read_csv('src/adult.csv')

adult_df['TPR_FEMALE_VP'] = (adult_df['income'] == ' >50K') & (adult_df['sex'] == ' Female')
adult_df['TPR_MALE_VP'] = (adult_df['income'] == ' >50K') & (adult_df['sex'] == ' Male')
adult_df['TPR_FEMALE_FP'] = (adult_df['income'] != ' >50K') & (adult_df['sex'] == ' Female')
adult_df['TPR_MALE_FP'] = (adult_df['income'] != ' >50K') & (adult_df['sex'] == ' Male')
adult_df = adult_df[['TPR_FEMALE_VP','TPR_MALE_VP','TPR_FEMALE_FP','TPR_MALE_FP']]
adult_df = adult_df.to_numpy()
print(np.sum(adult_df,axis = 0),np.sum(adult_df))