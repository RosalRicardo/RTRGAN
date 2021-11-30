import train

def train_plot(df, epochs, batchsize, fair_epochs, lamda, nu, S, Y, S_under, Y_desire):
    generator, critic, ohe, scaler, data_train, data_test, input_dim, critic_losses, gen_losses = train.train(df, S, Y, S_under, Y_desire, epochs, batchsize, fair_epochs, lamda, nu)
    return generator, critic, ohe, scaler, data_train, data_test, input_dim, critic_losses, gen_losses

    