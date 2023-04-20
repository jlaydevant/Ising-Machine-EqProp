'''
File to plot train_error and test_error over epochs
@author: Jeremie Laydevant
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob


def plot_error(dataframe, idx):
    '''
    Plot train and test error over epochs
    '''
    ave_train_error_tab = np.array(dataframe['Ave_Train_Error'].values.tolist())
    ave_test_error_tab = np.array(dataframe['Ave_Test_Error'].values.tolist())

    single_train_error_tab = np.array(dataframe['Single_Train_Error'].values.tolist())
    single_test_error_tab = np.array(dataframe['Single_Test_Error'].values.tolist())

    # plt.figure()
    plt.plot(ave_train_error_tab, label = 'ave train error #' + str(idx))
    plt.plot(ave_test_error_tab, label = 'ave test error #' + str(idx))

    plt.plot(single_train_error_tab, '--', label = 'single train error #' + str(idx))
    plt.plot(single_test_error_tab, '--', label = 'single test error #' + str(idx))

    plt.ylabel('Error (%)')
    plt.xlabel('Epochs')

    plt.title('Train and Test error (averaged and single neuron)')
    plt.legend()

    return ave_train_error_tab, ave_test_error_tab, single_train_error_tab, single_test_error_tab


def plot_alpha(dataframe, idx):
    '''
    Plot scaling factors over epochs
    '''
    for k in range(10):
        try:
            alpha_tab = dataframe['Alpha_' + str(k)].to_numpy()
            plt.plot(alpha_tab, label = 'Alpha_' + str(k))
        except:
            pass

    plt.ylabel('Alpha')
    plt.xlabel('Epochs')

    plt.title('Scaling factors')
    plt.legend()

    return 0


def plot_loss(dataframe, idx):
    '''
    Plot train and test error over epochs
    '''
    train_loss_tab = np.array(dataframe['Train_Loss'].values.tolist())
    test_loss_tab = np.array(dataframe['Test_Loss'].values.tolist())

    # plt.figure()
    plt.plot(train_loss_tab, label = 'train loss #' + str(idx))
    plt.plot(test_loss_tab, label = 'test loss #' + str(idx))

    plt.ylabel('Error (%)')
    plt.xlabel('Epochs')

    plt.title('Train and Test loss')
    plt.legend()

    return train_loss_tab, test_loss_tab


def plot_mean_error(store_ave_train_error, store_ave_test_error, store_single_train_error, store_single_test_error):
    '''
    Plot mean train & test error with +/- std
    '''
    try:
        store_ave_train_error, store_ave_test_error = np.array(store_ave_train_error), np.array(store_ave_test_error)
        mean_ave_train, mean_ave_test = np.mean(store_ave_train_error, axis = 0), np.mean(store_ave_test_error, axis = 0)
        std_ave_train, std_ave_test = np.std(store_ave_train_error, axis = 0), np.std(store_ave_test_error, axis = 0)

        store_single_train_error, store_single_test_error = np.array(store_single_train_error), np.array(store_single_test_error)
        mean_single_train, mean_single_test = np.mean(store_single_train_error, axis = 0), np.mean(store_single_test_error, axis = 0)
        std_single_train, std_single_test = np.std(store_single_train_error, axis = 0), np.std(store_single_test_error, axis = 0)

        epochs = np.arange(0, len(store_ave_test_error[0]))

        plt.figure()
        plt.plot(epochs, mean_ave_train, label = 'mean_ave_train_error')
        plt.fill_between(epochs, mean_ave_train - std_ave_train, mean_ave_train + std_ave_train, facecolor = '#b9f3f3')

        plt.plot(epochs, mean_ave_test, label = 'mean_ave_test_error')
        plt.fill_between(epochs, mean_ave_test - std_ave_test, mean_ave_test + std_ave_test, facecolor = '#fadcb3')

        plt.plot(epochs, mean_single_train, '--', label = 'mean_single_train_error')
        plt.fill_between(epochs, mean_single_train - std_single_train, mean_single_train + std_single_train, facecolor = '#b9f3f3')

        plt.plot(epochs, mean_single_test, '--', label = 'mean_single_test_error')
        plt.fill_between(epochs, mean_single_test - std_single_test, mean_single_test + std_single_test, facecolor = '#fadcb3')

        plt.ylabel('Error (%)')
        plt.xlabel('Epochs')
        plt.title('Mean train and Test error with std')
        plt.legend()

    except:
        pass

    return 0


def plot_mean_loss(store_train_loss, store_test_loss):
    '''
    Plot mean train & test loss with +/- std
    '''
    try:
        store_train_loss, store_test_loss = np.array(store_train_loss), np.array(store_test_loss)
        mean_train, mean_test = np.mean(store_train_loss, axis = 0), np.mean(store_test_loss, axis = 0)
        std_train, std_test = np.std(store_train_loss, axis = 0), np.std(store_test_loss, axis = 0)
        epochs = np.arange(0, len(store_test_loss[0]))
        plt.figure()
        plt.plot(epochs, mean_train, label = 'mean_train_loss')
        plt.fill_between(epochs, mean_train - std_train, mean_train + std_train, facecolor = '#b9f3f3')

        plt.plot(epochs, mean_test, label = 'mean_test_loss')
        plt.fill_between(epochs, mean_test - std_test, mean_test + std_test, facecolor = '#fadcb3')

        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.title('Mean train and Test loss with std')
        plt.legend()

    except:
        pass

    return 0

if __name__ == '__main__':
    if os.name != 'posix':
        path = "\\\\?\\" + os.getcwd()
        prefix = '\\'

    else:
        path = os.getcwd()
        prefix = '/'

    files = glob.glob('*')
    store_ave_train_error, store_single_train_error, store_train_loss = [], [], []
    store_ave_test_error, store_single_test_error, store_test_loss = [], [], []

    plt.figure()
    for idx, simu in enumerate(files):
        name, extension = os.path.splitext(simu)
        if not extension == '.py':
            DATAFRAME = pd.read_csv(path + prefix + simu + prefix + 'results.csv', sep = ',', index_col = 0)
            #plot error
            ave_train_error_tab, ave_test_error_tab, single_train_error_tab, single_test_error_tab = plot_error(DATAFRAME, name)
            store_ave_train_error.append(ave_train_error_tab)
            store_ave_test_error.append(ave_test_error_tab)
            store_single_train_error.append(single_train_error_tab)
            store_single_test_error.append(single_test_error_tab)

        else:
            pass

    plt.figure()
    for idx, simu in enumerate(files):
        name, extension = os.path.splitext(simu)
        if not extension == '.py':
            DATAFRAME = pd.read_csv(path + prefix + simu + prefix + 'results.csv', sep = ',', index_col = 0)
            #plot loss
            train_loss_tab, test_loss_tab = plot_loss(DATAFRAME, name)
            store_train_loss.append(train_loss_tab)
            store_test_loss.append(test_loss_tab)
        else:
            pass

    # plt.figure()
    # for idx, simu in enumerate(files):
    #     name, extension = os.path.splitext(simu)
    #     if not extension == '.py':
    #         DATAFRAME = pd.read_csv(path + prefix + simu + prefix + 'results.csv', sep = ',', index_col = 0)
    #         plot_alpha(DATAFRAME, idx)
    #     else:
    #         pass

    plot_mean_error(store_ave_train_error, store_ave_test_error, store_single_train_error, store_single_test_error)
    plot_mean_loss(store_train_loss, store_test_loss)


    plt.show()