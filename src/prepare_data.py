import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import get_ohe_data

def prepare_data(df, batch_size, S, Y, S_under, Y_desire):
    #df = pd.concat([df_train, df_test], axis=0)

    ohe, scaler, discrete_columns, continuous_columns, df_transformed, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index = get_ohe_data.get_ohe_data(df, S, Y, S_under, Y_desire)

    input_dim = df_transformed.shape[1]

    #from sklearn.model_selection import train_test_split
    #################
    X_train, X_test = train_test_split(df_transformed,test_size=0.1, shuffle=True) #random_state=10)
    #X_train = df_transformed[:df_train.shape[0],:]
    #X_test = df_transformed[df_train.shape[0]:,:]

    data_train = X_train.copy()
    data_test = X_test.copy()


    data = torch.from_numpy(data_train).float()


    train_ds = TensorDataset(data)
    train_dl = DataLoader(train_ds, batch_size = batch_size, drop_last=True)
    return ohe, scaler, input_dim, discrete_columns, continuous_columns ,train_dl, data_train, data_test, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index