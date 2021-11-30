import numpy as np
import pandas as pd
import torch
import torch.nn.functional as f
from torch import nn

class FairLossFunc(nn.Module):
    def __init__(self, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index):
        super(FairLossFunc, self).__init__()
        self._S_start_index = S_start_index
        self._Y_start_index = Y_start_index
        self._underpriv_index = underpriv_index
        self._priv_index = priv_index
        self._undesire_index = undesire_index
        self._desire_index = desire_index

    def forward(self, x, crit_fake_pred, lamda, nu):
        G = x[:, self._S_start_index:self._S_start_index + 2]
        # print(x[0,64])
        I = x[:, self._Y_start_index:self._Y_start_index + 2]
        # disp = (torch.mean(G[:,1]*I[:,1])/(x[:,65].sum())) - (torch.mean(G[:,0]*I[:,0])/(x[:,64].sum()))
        # disp = -1.0 * torch.tanh(torch.mean(G[:,0]*I[:,1])/(x[:,64].sum()) - torch.mean(G[:,1]*I[:,1])/(x[:,65].sum()))
        # gen_loss = -1.0 * torch.mean(crit_fake_pred)
        disp = -1.0 * lamda * (torch.mean(G[:, self._underpriv_index] * I[:, self._desire_index]) / (
            x[:, self._S_start_index + self._underpriv_index].sum()) - torch.mean(
            G[:, self._priv_index] * I[:, self._desire_index]) / (
                                   x[:, self._S_start_index + self._priv_index].sum())) - 1.0 * torch.mean(
            crit_fake_pred)
        # teste de oversampling de grupos sensiveis
        taxa_missing_data = nu * (torch.mean(torch.abs((x[:, self._S_start_index + self._priv_index].sum() - 
            x[:, self._S_start_index + self._underpriv_index].sum())) / x[:, self._S_start_index + self._underpriv_index].size(axis=0)))
        
        #novo disp
        disp = disp + taxa_missing_data

        # print(disp)
        return disp