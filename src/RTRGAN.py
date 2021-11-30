import torch
import torch.nn.functional as f
from torch import nn
import pandas as pd
import numpy as np
import get_ohe_data
import get_original_data

import classes
import train
import train_plot
import multiple_runs
import print2file
import criterion


from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

epochs = 2
batch_size = 64
fair_epochs = 1
lamda = 0.5
nu = 0.5
test_path = 'tests/mdr/'
fake_name = 'fake_data_wgan_mdr_'

size_fake = 1000
nu = 0.1
S = "sex"
Y = "income"
S_under = " Female"
Y_desire = ">50K"
df = pd.read_csv("src/adult.csv")

df[S] = df[S].astype(object)
df[Y] = df[Y].astype(object)

display_step = 50

outfile = "results_wgan_mdr_roc.log"
multiple_runs.multiple_runs(num_trainings=4, df=df, epochs=epochs, batchsize=batch_size, fair_epochs=fair_epochs, lamda=lamda, nu=nu,
 test_path=test_path,fake_name=fake_name, outFile=outfile, S=S, Y=Y, S_under=S_under, Y_desire=Y_desire)
