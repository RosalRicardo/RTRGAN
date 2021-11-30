import pandas as pd
import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import metrics
import print2file
import train_plot
import get_original_data

device = torch.device("cuda:0" if (torch.cuda.is_available() and 1 > 0) else "cpu")

def multiple_runs(num_trainings, df, epochs, batchsize, fair_epochs, lamda, nu, test_path, fake_name, outFile, S, Y, S_under, Y_desire):
    print(lamda)
    for i in range(num_trainings):
        if i == 0:
            first_line = "num_trainings: %d, num_epochs: %d, batchsize: %d, fair_epochs:%d, lamda:%f, nu:%f" % (
            num_trainings, epochs, batchsize, fair_epochs, lamda, nu)
            print(first_line)
            print2file.print2file(first_line, outFile)
            # print2file(buf, outFile)

            second_line = "train_idx ,accuracy_original, accuracy_generated, difference_accuracy, f1_original, f1_generated, difference_f1, demographic_parity_data, demographic_parity_classifier, roc_threshold, nu"
            print(second_line)
            print2file.print2file(second_line, outFile)

        generator, critic, ohe, scaler, data_train, data_test, input_dim, critic_losses, gen_losses = train_plot.train_plot(df=df, epochs=epochs, batchsize=batchsize,
                                                                           fair_epochs=fair_epochs, lamda=lamda, nu=nu, S=S, Y=Y, S_under=S_under, Y_desire=Y_desire)
        data_train_x = data_train[:, :-2]
        data_train_y = np.argmax(data_train[:, -2:], axis=1)

        data_test_x = data_test[:, :-2]
        data_test_y = np.argmax(data_test[:, -2:], axis=1)
        

        df_generated = generator(torch.randn(size=(32561, input_dim), device=device)).cpu().detach().numpy()
        df_generated_x = df_generated[:, :-2]
        df_generated_y = np.argmax(df_generated[:, -2:], axis=1)

        
        clf = DecisionTreeClassifier()
        clf = clf.fit(data_train_x, data_train_y)
        
        # print("accuracy original = {}".format(accuracy_score(clf.predict(data_test_x), data_test_y)))
        accuracy_original = accuracy_score(clf.predict(data_test_x), data_test_y)
        
        f1_original = f1_score(clf.predict(data_test_x), data_test_y)

        
        clf = DecisionTreeClassifier()
        clf = clf.fit(df_generated_x, df_generated_y)
        
        # print("accuracy generated = {}".format(accuracy_score(clf.predict(data_test_x), data_test_y)))
        accuracy_generated = accuracy_score(clf.predict(data_test_x), data_test_y)

        
        # curva ROC
        fpr, tpr, thresholds = metrics.roc_curve(data_test_y, clf.predict(data_test_x))
        roc_curve = pd.DataFrame(np.c_[fpr,tpr])

        f1_generated = f1_score(clf.predict(data_test_x), data_test_y)

        difference_accuracy = accuracy_original - accuracy_generated
        difference_f1 = f1_original - f1_generated

        female_mask = df_generated_x[:, 64] == 1
        male_mask = 1 - female_mask
        male_mask = male_mask == 1
        rich_mask = df_generated_y == 1
        female_odds = ((female_mask & rich_mask).sum()) / (female_mask.sum())
        male_odds = ((male_mask & rich_mask).sum()) / (male_mask.sum())
        demographic_parity_data = female_odds - male_odds

        predictions = clf.predict(data_test_x)
        female_mask = data_test_x[:, 64] == 1
        male_mask = 1 - female_mask
        male_mask = male_mask == 1
        prediction_mask = predictions == 1
        female_odds = ((female_mask & prediction_mask).sum()) / (female_mask.sum())
        male_odds = ((male_mask & prediction_mask).sum()) / (male_mask.sum())
        demographic_parity_classifier = female_odds - male_odds

        # print("\nfairness original data: male_odds:{} \t female odds:{}".format(male_odds, female_odds))

        buf = '%d, %f, %f , %f, %f, %f, %f, %f, %f, %f, %f' % (
        i, accuracy_original, accuracy_generated, difference_accuracy, f1_original, f1_generated, difference_f1,
        demographic_parity_data, demographic_parity_classifier, thresholds[0], nu)
        
        critic_losses = pd.DataFrame(critic_losses)
        critic_losses.to_csv(test_path + 'critic_losses_WGAN_mdr_nu' + str(nu) + '_'+str(i)+'.csv',index=False)
        gen_losses = pd.DataFrame(gen_losses)
        gen_losses.to_csv(test_path + 'gen_losses_WGAN_mdr_nu' + str(nu) + '_'+str(i)+'.csv',index=False)
        roc_curve.to_csv(test_path + 'roc_nu01_'+str(i)+'.csv')

        # classificador para calculo da curva ROC (dados falsos)
        #fake_numpy_array = generator(torch.randn(size=(size_fake, input_dim), device=device)).cpu().detach().numpy()
        fake_df = get_original_data.get_original_data(df_generated, df, ohe, scaler)
        fake_df = fake_df[df.columns]
        fake_df['predict'] = clf.predict(df_generated_x)
        fake_df.to_csv(test_path + fake_name + str(nu) + '_'+str(i)+'.csv', index=False)

        print(buf)
        print2file.print2file(buf, outFile)
