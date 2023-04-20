# coding: utf-8
import os
import os.path
import datetime
import time
import numpy as np
import torch.nn as nn
from scipy import*
from copy import*
import matplotlib.pyplot as plt
import sys
import pickle
import pandas as pd
import shutil
from tqdm import tqdm
from Network import*


def train_bin(net, args, train_loader, epoch, trackGrad = None, trackGradFilt = None, trackW = None, trackChanges = False):
    '''
    Function to train the network for 1 epoch
    '''
    net.train()
    net.epoch = epoch+1
    criterion = nn.MSELoss(reduction = 'sum')
    ave_falsePred, single_falsePred, loss_loc = 0, 0, 0

    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):

        if args.randomBeta:
            net.beta = torch.sign(torch.randn(1)) * args.beta

        s = net.initHidden(args, data)

        if net.cuda:
            targets = targets.to(net.device)
            net.beta = net.beta.to(net.device)
            for i in range(len(s)):
                s[i] = s[i].to(net.device)

        #free phase
        s = net.forward(args, s)
        seq = s.copy()

        #loss
        loss = (1/(2*s[0].size(0)))*criterion(s[0], targets)
        loss_loc += loss

        #nudged phase
        s = net.forward(args, s, target = targets, beta = net.beta)

        #update and track the weights of the network
        net.updateWeight(epoch, s, seq, args)

        #compute averaged error over the sub-classes
        pred_ave = torch.stack([item.sum(1) for item in seq[0].split(args.expand_output, dim = 1)], 1)/args.expand_output
        targets_red = torch.stack([item.sum(1) for item in targets.split(args.expand_output, dim = 1)], 1)/args.expand_output
        ave_falsePred += (torch.argmax(targets_red, dim = 1) != torch.argmax(pred_ave, dim = 1)).int().sum(dim=0)

        # #compute error computed on the first neuron of each sub class
        pred_single = torch.stack([item[:,0] for item in seq[0].split(args.expand_output, dim = 1)], 1)
        single_falsePred += (torch.argmax(targets_red, dim = 1) != torch.argmax(pred_single, dim = 1)).int().sum(dim=0)

        del s, seq, data, targets, pred_ave, pred_single
        net.beta = args.beta


    ave_train_error = (ave_falsePred.float() / float(len(train_loader.dataset))) * 100
    single_train_error = (single_falsePred.float() / float(len(train_loader.dataset))) * 100
    total_loss = loss_loc/ float(batch_idx)

    return ave_train_error, single_train_error, total_loss


def test_bin(net, args, test_loader):
    '''
    Function to test the network
    '''
    net.eval()
    criterion = nn.MSELoss(reduction = 'sum')
    ave_falsePred, single_falsePred, loss_loc = 0, 0, 0

    for batch_idx, (data, targets) in enumerate(test_loader):

        s = net.initHidden(args, data)

        if net.cuda:
            targets = targets.to(net.device)
            for i in range(len(s)):
                s[i] = s[i].to(net.device)

        #free phase
        s = net.forward(args, s)

        #loss
        loss_loc += (1/(2*s[0].size(0)))*criterion(s[0], targets*net.neuronMax)

        #compute averaged error over the sub_classses
        pred_ave = torch.stack([item.sum(1) for item in s[0].split(args.expand_output, dim = 1)], 1)/args.expand_output
        targets_red = torch.stack([item.sum(1) for item in targets.split(args.expand_output, dim = 1)], 1)/args.expand_output
        ave_falsePred += (torch.argmax(targets_red, dim = 1) != torch.argmax(pred_ave, dim = 1)).int().sum(dim=0)

        # #compute error computed on the first neuron of each sub class
        pred_single = torch.stack([item[:,0] for item in s[0].split(args.expand_output, dim = 1)], 1)
        single_falsePred += (torch.argmax(targets_red, dim = 1) != torch.argmax(pred_single, dim = 1)).int().sum(dim=0)

    ave_test_error = (ave_falsePred.float() / float(len(test_loader.dataset))) * 100
    single_test_error = (single_falsePred.float() / float(len(test_loader.dataset))) * 100
    test_loss = loss_loc/ float(batch_idx)

    return ave_test_error, single_test_error, test_loss


def initDataframe(path, args, net, dataframe_to_init = 'results.csv'):
    '''
    Initialize a dataframe with Pandas so that parameters are saved
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    if os.path.isfile(path + dataframe_to_init):
        dataframe = pd.read_csv(path + dataframe_to_init, sep = ',', index_col = 0)
    else:
        columns_header = ['Ave_Train_Error','Ave_Test_Error','Single_Train_Error','Single_Test_Error', 'Train_Loss','Test_Loss' ]

        dataframe = pd.DataFrame({},columns = columns_header)
        dataframe.to_csv(path + prefix + 'results.csv')

    return dataframe


def updateDataframe(BASE_PATH, args, dataframe, net, ave_train_error, ave_test_error, single_train_error, single_test_error, train_loss, test_loss):
    '''
    Add data to the pandas dataframe
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    data = [ave_train_error, ave_test_error, single_train_error, single_test_error, train_loss, test_loss]

    new_data = pd.DataFrame([data],index=[1],columns=dataframe.columns)

    dataframe = pd.concat([dataframe, new_data], axis=0)
    dataframe.to_csv(BASE_PATH + prefix +'results.csv')

    return dataframe


def createPath(args):
    '''
    Create path to save data
    '''

    if os.name != 'posix':
        BASE_PATH = "\\\\?\\" + os.getcwd()
        prefix = '\\'

    else:
        BASE_PATH = os.getcwd()
        prefix = '/'

    BASE_PATH += prefix + 'DATA-0'

    BASE_PATH += prefix + datetime.datetime.now().strftime("%Y-%m-%d")


    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)

    filePath = shutil.copy('plotFunction.py', BASE_PATH)

    files = os.listdir(BASE_PATH)

    if 'plotFunction.py' in files:
        files.pop(files.index('plotFunction.py'))

    if not files:
        BASE_PATH = BASE_PATH + prefix + 'S-1'
    else:
        tab = []
        if '.DS_Store' in files:
            files.pop(files.index('.DS_Store'))
        for names in files:
            tab.append(int(names.split('-')[1]))
        BASE_PATH += prefix + 'S-' + str(max(tab)+1)

    try:
        os.mkdir(BASE_PATH)
    except:
        pass
    name = BASE_PATH.split(prefix)[-1]


    return BASE_PATH, name


def saveHyperparameters(args, net, BASE_PATH):
    '''
    Save all hyperparameters in the path provided
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    f = open(BASE_PATH + prefix + 'Hyperparameters.txt', 'w')
    f.write('Binary Equilibrium Propagation \n')
    f.write('   Parameters of the simulation \n ')
    f.write('\n')

    for key in args.__dict__:
        f.write(key)
        f.write(': ')
        if key == "gradThreshold":
            f.write(str(net.threshold))
        else:
            f.write(str(args.__dict__[key]))
        f.write('\n')

    f.close()


def generate_digits(args):
    '''
    Generate the dataloaders for digits dataset
    '''
    digits = load_digits()

    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.1, random_state=10, shuffle=True)
    normalisation = 8
    x_train, x_test = x_train / normalisation, x_test / normalisation

    train_data = DefineDataset(x_train, labels=y_train, target_transforms=ReshapeTransformTarget(10, args))
    test_data = DefineDataset(x_test, labels=y_test, target_transforms=ReshapeTransformTarget(10, args))

    ## Data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    return train_loader, test_loader


def generate_mnist(args):
    '''
    Generate mnist dataloaders - 1000 training images, 100 testing images
    '''
    N_class = 10
    N_data = args.N_data
    N_data_test = args.N_data_test

    with torch.no_grad():
        if args.data_augmentation:
            transforms_train=[torchvision.transforms.ToTensor(), torchvision.transforms.RandomAffine(10, translate=(0.04, 0.04), scale=None, shear=None, interpolation=torchvision.transforms.InterpolationMode.NEAREST, fill=0), ReshapeTransform((-1,))]
        else:
            transforms_train=[torchvision.transforms.ToTensor(), ReshapeTransform((-1,))]

        transforms_test=[torchvision.transforms.ToTensor(), ReshapeTransform((-1,))]

        #Training data
        mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                                transform=torchvision.transforms.Compose(transforms_train),
                                                target_transform=ReshapeTransformTarget(10, args))

        mnist_train_data, mnist_train_targets, comp = torch.empty(N_data,28,28,dtype=mnist_train.data.dtype), torch.empty(N_data,dtype=mnist_train.targets.dtype), torch.zeros(N_class)
        idx_0, idx_1 = 0, 0
        while idx_1 < N_data:
            class_data = mnist_train.targets[idx_0]
            if comp[class_data] < int(N_data/N_class):
                mnist_train_data[idx_1,:,:] = mnist_train.data[idx_0,:,:].clone()
                mnist_train_targets[idx_1] = class_data.clone()
                comp[class_data] += 1
                idx_1 += 1
            idx_0 += 1

        mnist_train.data, mnist_train.targets = mnist_train_data, mnist_train_targets

        train_loader = torch.utils.data.DataLoader(mnist_train, batch_size = args.batch_size, shuffle=True)

        #Testing data
        mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                                transform=torchvision.transforms.Compose(transforms_test),
                                                target_transform=ReshapeTransformTarget(10, args))

        mnist_test_data, mnist_test_targets, comp = torch.empty(N_data_test,28,28,dtype=mnist_test.data.dtype), torch.empty(N_data_test,dtype=mnist_test.targets.dtype), torch.zeros(N_class)
        idx_0, idx_1 = 0, 0
        while idx_1 < N_data_test:
            class_data = mnist_test.targets[idx_0]
            if comp[class_data] < int(N_data_test/N_class):
                mnist_test_data[idx_1,:,:] = mnist_test.data[idx_0,:,:].clone()
                mnist_test_targets[idx_1] = class_data.clone()
                comp[class_data] += 1
                idx_1 += 1
            idx_0 += 1

        mnist_test.data, mnist_test.targets = mnist_test_data, mnist_test_targets

        test_loader = torch.utils.data.DataLoader(mnist_test, batch_size = 1, shuffle=False)

        return train_loader, test_loader, mnist_train

