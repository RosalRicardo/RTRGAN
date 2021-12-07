import torch
import torch.nn.functional as f
from torch import nn
import pandas as pd
import numpy as np
import classes
import time
import prepare_data
import criterion

device = torch.device("cuda:0" if (torch.cuda.is_available() and 1 > 0) else "cpu")

def train(df, S, Y, S_under, Y_desire, epochs=500, batch_size=64, fair_epochs=10, lamda=0.5, nu=0.5):
    ohe, scaler, input_dim, discrete_columns, continuous_columns, train_dl, data_train, data_test, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index = prepare_data.prepare_data(
        df, batch_size, S, Y, S_under, Y_desire)

    generator = classes.Generator(input_dim, continuous_columns, discrete_columns).to(device)
    critic = classes.Critic(input_dim).to(device)
    second_critic = classes.DISPLoss(S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index).to(
        device)
    third_critic = classes.RTRLoss(S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index).to(
        device)
    
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    gen_optimizer_fair = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    crit_optimizer = torch.optim.Adam(critic.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # loss = nn.BCELoss()
    critic_losses = []
    gen_losses = []
    cur_step = 0
    for i in range(epochs):
        # j = 0
        start = time.time()
        print("epoch {}".format(i + 1))
        ############################
        if i + 1 <= (epochs - fair_epochs):
            print("training for accuracy")
        if (i + 1 <= epochs - (fair_epochs/2)) and (i + 1 > (epochs - fair_epochs)):
            print("training for fairness (DISP)")
        if i + 1 > (epochs - fair_epochs/2):
            print("training for fairness (RTR)")
        for data in train_dl:
            data[0] = data[0].to(device)
            # j += 1
            loss_of_epoch_G = 0
            loss_of_epoch_D = 0
            crit_repeat = 4
            mean_iteration_critic_loss = 0
            mean_iteration_gen_loss = 0
            for k in range(crit_repeat):
                # training the critic
                crit_optimizer.zero_grad()
                fake_noise = torch.randn(size=(batch_size, input_dim), device=device).float()
                fake = generator(fake_noise)

                crit_fake_pred = critic(fake.detach())
                crit_real_pred = critic(data[0])

                epsilon = torch.rand(batch_size, input_dim, device=device, requires_grad=True)
                gradient = criterion.get_gradient(critic, data[0], fake.detach(), epsilon)
                gp = criterion.gradient_penalty(gradient)

                crit_loss = criterion.get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda=10)

                mean_iteration_critic_loss += crit_loss.item() / crit_repeat
                crit_loss.backward(retain_graph=True)
                crit_optimizer.step()
            #############################
            if cur_step > 50:
                critic_losses += [mean_iteration_critic_loss]

            #############################
            if i + 1 <= (epochs - fair_epochs):
                # training the generator for accuracy
                gen_optimizer.zero_grad()
                fake_noise_2 = torch.randn(size=(batch_size, input_dim), device=device).float()
                fake_2 = generator(fake_noise_2)
                crit_fake_pred = critic(fake_2)

                gen_loss = criterion.get_gen_loss(crit_fake_pred)
                mean_iteration_gen_loss += gen_loss.item()
                gen_loss.backward()

                # Update the weights
                gen_optimizer.step()

            #############################
                if cur_step > 50:
                    gen_losses += [mean_iteration_gen_loss]

            #############################  
            # Training for disparete impact
            ###############################
            if (i + 1 <= epochs - (fair_epochs/2)) and (i + 1 > (epochs - fair_epochs)):
                # training the generator for fairness
                gen_optimizer_fair.zero_grad()
                fake_noise_2 = torch.randn(size=(batch_size, input_dim), device=device).float()
                fake_2 = generator(fake_noise_2)

                crit_fake_pred = critic(fake_2)

                gen_fair_loss = third_critic(fake_2, crit_fake_pred, lamda, nu)
                mean_iteration_gen_loss += gen_fair_loss.item()
                gen_fair_loss.backward()
                gen_optimizer_fair.step()

            #############################  
            # Training for representativeness
            ###############################
            if i + 1 > (epochs - (fair_epochs/2)):
                # training the generator for fairness
                gen_optimizer_fair.zero_grad()
                fake_noise_2 = torch.randn(size=(batch_size, input_dim), device=device).float()
                fake_2 = generator(fake_noise_2)

                crit_fake_pred = critic(fake_2)

                gen_fair_loss = third_critic(fake_2, crit_fake_pred, lamda, nu)
                mean_iteration_gen_loss += gen_fair_loss.item()
                gen_fair_loss.backward()
                gen_optimizer_fair.step()

            #############################
                if cur_step > 50:
                    gen_losses += [mean_iteration_gen_loss]

            #############################  
            """
            # Keep track of the average generator loss
            #################################
            if cur_step > 50:
                if i + 1 <= (epochs - fair_epochs):
                    generator_losses += [gen_loss.item()]
                if i + 1 > (epochs - fair_epochs):
                    generator_losses += [gen_fair_loss.item()]

                    # print("cr step: {}".format(cur_step))
            if cur_step % display_step == 0 and cur_step > 0:
                gen_mean = sum(generator_losses[-display_step:]) / display_step
                crit_mean = sum(critic_losses[-display_step:]) / display_step
                print("Step {}: Generator loss: {}, critic loss: {}".format(cur_step, gen_mean, crit_mean))
                step_bins = 20
                num_examples = (len(generator_losses) // step_bins) * step_bins
                plt.plot(
                    range(num_examples // step_bins),
                    torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Generator Loss"
                )
                plt.plot(
                    range(num_examples // step_bins),
                    torch.Tensor(critic_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Critic Loss"
                )
                plt.legend()
                plt.show()
	    """
            end = time.time()
            cur_step += 1
    
    torch.save(generator.state_dict(), 'src/modelo/generator.pth')
    torch.save(critic.state_dict(), 'src/modelo/critic.pth')
    torch.save(second_critic.state_dict(), 'src/modelo/DISPLoss.pth')
    torch.save(third_critic.state_dict(), 'src/modelo/RTRLoss.pth')

    return generator, critic, ohe, scaler, data_train, data_test, input_dim, critic_losses, gen_losses