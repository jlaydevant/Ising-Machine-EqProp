import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import matplotlib

font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)
plt.rcParams['axes.linewidth'] = 2


def plot_error(dataframe, idx, color):

    train_error_tab = np.array(dataframe['QPU_Train_Accuracy'].values.tolist())
    plt.plot(train_error_tab, color = color, label = 'exp-' + str(idx.split('-')[-1]))
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epochs')
    plt.legend()

    return train_error_tab, test_error_tab


def plot_loss(dataframe, idx, color):

    train_loss_tab = np.array(dataframe['Train_Loss'].values.tolist())
    plt.plot(train_loss_tab, color = color, label = 'exp-' + str(idx.split('-')[-1]))
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    return train_loss_tab, test_loss_tab


def plot_mean_error(store_ave_train_error, store_ave_test_error):
    '''
    Plot mean train & test error with +/- std
    '''
    try:
        store_ave_train_error = np.array(store_ave_train_error)
        mean_ave_train = np.mean(store_ave_train_error, axis = 0)
        std_ave_train = np.std(store_ave_train_error, axis = 0)
        epochs = np.arange(0, len(store_ave_train_error[0]))

        plt.figure(figsize= (9,6))
        plt.plot(epochs, mean_ave_train, color = "#DA355E",label = 'Mean Train Accuracy D-Wave')
        plt.fill_between(epochs, mean_ave_train - std_ave_train, (mean_ave_train + std_ave_train).clip(0,100), facecolor = '#DA355E', alpha = 0.25)

        plt.ylabel('Accuracy (%)')
        plt.xlabel('Epochs')
        plt.legend()
        plt.grid()

    except:
        pass

    return 0


def plot_mean_loss(store_train_loss, store_test_loss):
    '''
    Plot mean train & test loss with +/- std
    '''
    try:
        store_train_loss = np.array(store_train_loss)
        mean_train = np.mean(store_train_loss, axis = 0)
        std_train = np.std(store_train_loss, axis = 0)
        epochs = np.arange(0, len(store_train_loss[0]))

        fig = plt.figure(figsize= (9,6))
        ax = fig.add_axes([0.1, 0.15, 0.8, 0.8])

        ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='off')
        ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')

        plt.plot(epochs, mean_train, color = "#DA355E", label = 'Mean Train Loss D-Wave')
        plt.fill_between(epochs, (mean_train - std_train).clip(0), mean_train + std_train, facecolor = '#DA355E', alpha = 0.25)

        plt.grid()
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
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
    store_train_error, store_train_loss = [], []
    store_test_error, store_test_loss = [], []

    colormap = plt.cm.RdPu
    colors = [colormap(i) for i in np.linspace(0, 1, 6)]

    print(files)
    for idx, simu in enumerate(files):
        name, extension = os.path.splitext(simu)
        if extension == '.py':
            files.remove(simu)
    files = sorted(files, key = lambda x: (int(x.split('-')[-1])))
    print(files)

    plt.figure(figsize= (9,6))
    for idx, simu in enumerate(files):
        name, extension = os.path.splitext(simu)
        DATAFRAME = pd.read_csv(path + prefix + simu + prefix + 'results.csv', sep = ',', index_col = 0)
        #plot error
        train_error_tab, test_error_tab = plot_error(DATAFRAME, name, color = colors[-(idx+1)])
        store_train_error.append(train_error_tab)
        store_test_error.append(test_error_tab)


    plt.figure(figsize= (9,6))
    for idx, simu in enumerate(files):
        name, extension = os.path.splitext(simu)
        DATAFRAME = pd.read_csv(path + prefix + simu + prefix + 'results.csv', sep = ',', index_col = 0)
        #plot loss
        train_loss_tab, test_loss_tab = plot_loss(DATAFRAME, name, color = colors[-(idx+1)])
        store_train_loss.append(train_loss_tab)
        store_test_loss.append(test_loss_tab)

    print(train_error_tab)
    plot_mean_error(store_train_error, store_test_error)
    plot_mean_loss(store_train_loss, store_test_loss)


    plt.show()