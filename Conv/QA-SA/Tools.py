import os
import os.path
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
from scipy import*
from copy import*
import sys
import pandas as pd
import shutil
from tqdm import tqdm
import torchvision
import glob
import dimod
import matplotlib.pyplot as plt
import pickle

#================= DATA GENERATION ===================================================
class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class ReshapeTransformTarget:
    def __init__(self, args, number_classes):
        self.number_classes = number_classes
        self.args = args

    def __call__(self, target):
        target=torch.tensor(target).unsqueeze(0).unsqueeze(1)
        target_onehot = -1*torch.ones((1, self.number_classes))
        return target_onehot.scatter_(1, target.long(), 1).repeat_interleave(self.args.expand_output).squeeze(0)
        
class DefineDataset():
    def __init__(self, images, labels=None, transforms=None, target_transforms=None):
        self.x = images
        self.y = labels
        self.transforms = transforms
        self.target_transforms = target_transforms

    def __getitem__(self, i):
        data = self.x[i, :]
        target = self.y[i]

        if self.transforms:
            data = self.transforms(data)

        if self.target_transforms:
            target = self.target_transforms(target)

        if self.y is not None:
            return (data, target)
        else:
            return data

    def __len__(self):
        return (len(self.x))
    

def generate_patterns(args):
    '''
    Generate 2 patterns which are orthogonal diagonals in a 3x3 square
    --> already implemented for 3 patterns, bu 
    '''
    
    data_test = -1*torch.ones((2,1,3,3)) # 1: numbers of different patterns, 2: number of input channels: 1 here, 3,4: size of the input (3x3 here)
    for (idx1, idx2) in ((0,0),(1,1),(2,2)): data_test[0,:,idx1, idx2] = 1 #first pattern : diagonal right
    for (idx1, idx2) in ((0,2),(1,1),(2,0)): data_test[1,:,idx1, idx2] = 1 #second pattern : diagonal left

    target_test = torch.tensor([0,1])
    
    test_data = DefineDataset(data_test, labels=target_test, target_transforms=ReshapeTransformTarget(args, 2))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batchSize, shuffle=True)
    
    train_data = DefineDataset(data_train, labels=target_train, target_transforms=ReshapeTransformTarget(args, 2))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batchSize, shuffle=True)
    
    return train_loader, test_loader
    
    
#================= CONV ARCHITECTURES ===================================================
def createBQM(net, args, data, beta = 0, target = None, mode = None):

    #index of the patches of the images being convolved: (it is 2x2 patches as we use 2x2 kernels 
    image_idx = [[0,0],[0,1],[1,0],[1,1], [0,1],[0,2],[1,1],[1,2], [1,0],[1,1],[2,0],[2,1], [1,1],[1,2],[2,1],[2,2]]

    ## Biases
    bias_lim = args.bias_lim
    # inputs on the QPU - biases are large here in order to have the input qbits being clamped
    h = {idx: -2 if data[0][elm[0],elm[1]] > 0 else 2 for (idx, elm) in enumerate(image_idx)}
    
    comp = len(image_idx)
    
    #bias for neurons after convolution: no bias!
    h.update({item: 0 for item in net.index_before_pool})
    
    #no bias for neurons after pooling
    h.update({item: 0 for item in net.index_after_pool})
    
    #bias for output neurons - with or without nudge
    if target is not None:
        bias_nudge = -beta*target
        h.update({net.index_fc[k]: (bias_nudge[k] + net.fc[0].bias[k]).clip(-bias_lim,bias_lim).item() for k in range(len(net.fc[0].bias))})
    else:
        h.update({net.index_fc[k]: net.fc[0].bias[k].clip(-bias_lim,bias_lim).item() for k in range(len(net.fc[0].bias))})


    ## Couplings
    coupled_qbits = [[[[0,16],[4,20],[8,24],[12,28]],[[1,16],[5,20],[9,24],[13,28]],[[2,16],[6,20],[10,24],[14,28]],[[3,16],[7,20],[11,24],[15,28]]], 
                     [[[0,17],[4,21],[8,25],[12,29]],[[1,17],[5,21],[9,25],[13,29]],[[2,17],[6,21],[10,25],[14,29]],[[3,17],[7,21],[11,25],[15,29]]], 
                     [[[0,18],[4,22],[8,26],[12,30]],[[1,18],[5,22],[9,26],[13,30]],[[2,18],[6,22],[10,26],[14,30]],[[3,18],[7,22],[11,26],[15,30]]], 
                     [[[0,19],[4,23],[8,27],[12,31]],[[1,19],[5,23],[9,27],[13,31]],[[2,19],[6,23],[10,27],[14,31]],[[3,19],[7,23],[11,27],[15,31]]]]
    
    J = {}
    #convolution weights
    for k in range(args.convList[0]):
        comp=0
        for i in range(2):
            for j in range(2):
                J.update({(elm[0],elm[1]): net.conv[0].weight[k][0][i][j].item() for (idx,elm) in enumerate(coupled_qbits[k][comp])})
                comp+=1
        
    #pooling weights = 1/4
    coupled_qbits_pooling = [[[16,32],[20,32],[24,32],[28,32]],
                             [[17,33],[21,33],[25,33],[29,33]],
                             [[18,34],[22,34],[26,34],[30,34]],
                             [[19,35],[23,35],[27,35],[31,35]]]
                             
    for k in range(args.convList[0]):
        avg_pool_coef = -args.pool_coef
        J.update({(elm[0],elm[1]): avg_pool_coef for (idx,elm) in enumerate(coupled_qbits_pooling[k])})
    
    #fc weights
    for k in range(args.layersList[1]):
        for i in range(args.layersList[0]):
            J.update({(net.index_after_pool[k],net.index_fc[i]): net.fc[0].weight[i][k].item()})

    model = dimod.BinaryQuadraticModel.from_ising(h, J, 0)

    return model
   
   
def train(net, args, train_loader, sampler):
    '''
    function to train the network for 1 epoch
    '''
    exact_pred, exact_loss = 0, 0
    qpu_pred, qpu_loss = 0, 0

    with torch.no_grad():
        for idx, (DATA, TARGET) in enumerate(tqdm(train_loader)):
            store_seq = None
            store_s = None
            
            for k in range(DATA.size()[0]):
                data, target = DATA[k].numpy(), TARGET[k].numpy()

                ## Free phase
                model = createBQM(net, args, data)
                # QPU sampling
                qpu_seq = sampler.sample(model, num_reads = args.n_iter_free, chain_strength = args.chain_strength, auto_scale = args.auto_scale)

                ## Nudge phase: same system except bias for the output layer
                model = createBQM(net, args, data, beta = args.beta, target = target)
                reverse_schedule = [[0.0, 1.0], [10, args.frac_anneal_nudge], [20, 1]]
                reverse_anneal_params = dict(anneal_schedule=reverse_schedule,
                                    initial_state=qpu_seq.first.sample,
                                    reinitialize_state=True)

                if np.array_equal(qpu_seq.record["sample"][0].reshape(1,-1)[:,-args.layersList[1]:][0], target): 
                    qpu_s = qpu_seq
                else:
                    qpu_s = sampler.sample(model, num_reads = args.n_iter_nudge, chain_strength = args.chain_strength, auto_scale = args.auto_scale, **reverse_anneal_params)

                if store_seq is None:
                    store_seq = qpu_seq.record["sample"][0].reshape(1, qpu_seq.record["sample"][0].shape[0])
                    store_s = qpu_s.record["sample"][0].reshape(1, qpu_s.record["sample"][0].shape[0])
                else:
                    store_seq = np.concatenate((store_seq, qpu_seq.record["sample"][0].reshape(1, qpu_seq.record["sample"][0].shape[0])),0)
                    store_s = np.concatenate((store_s, qpu_s.record["sample"][0].reshape(1, qpu_s.record["sample"][0].shape[0])),0)

                del qpu_seq, qpu_s
                del data, target

            seq = net.sample_to_s(args, DATA, store_seq) 
            s   = net.sample_to_s(args, DATA, store_s)

            ## Compute loss and error for QPU sampling
            loss, pred = net.computeLossAcc(seq, TARGET, args, stage = 'training')

            qpu_pred += pred
            qpu_loss += loss

            net.updateParams(DATA, s, seq, args)
            net.sign_beta = 1

            del seq, s
            del DATA, TARGET

    return qpu_loss, qpu_pred


def initDataframe(path, dataframe_to_init = 'results.csv'):
    '''
    Initialize a dataframe with Pandas so that parameters are saved
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'
        
    print(path)
    print(dataframe_to_init)
    if os.path.isfile(path + dataframe_to_init):
        dataframe = pd.read_csv(path + dataframe_to_init, sep = ',', index_col = 0)
    else:
        columns_header = ['QPU_Train_Accuracy','Train_Loss']
        dataframe = pd.DataFrame({},columns = columns_header)
        dataframe.to_csv(path + prefix + 'results.csv')
    return dataframe


def updateDataframe(BASE_PATH, dataframe, train_error, train_loss):
    '''
    Add data to the pandas dataframe
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'
        
    data = [train_error, train_loss]


    new_data = pd.DataFrame([data],index=[1],columns=dataframe.columns)

    dataframe = pd.concat([dataframe, new_data], axis=0)
    dataframe.to_csv(BASE_PATH + prefix + 'results.csv')

    return dataframe

#=======================================================================================================
#=========================================== COMMONS ===================================================
#=======================================================================================================

def createPath(args):
    '''
    Create path to save data
    '''
    if os.name != 'posix':
        prefix = '\\'
        BASE_PATH = prefix + prefix + "?" + prefix + os.getcwd()
    else:
        prefix = '/'
        BASE_PATH = '' + os.getcwd()

    BASE_PATH += prefix + 'DATA' + str(args.dataset)

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
        for names in files:
            if names.split('.')[-1] != 'DS_Store':
                tab.append(int(names.split('-')[1]))
        BASE_PATH += prefix + 'S-' + str(max(tab)+1)
    

    return BASE_PATH


def saveHyperparameters(BASE_PATH, args):
    '''
    Save all hyperparameters in the path provided
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    f = open(BASE_PATH + prefix + 'Hyperparameters.txt', 'w')
 
    f.write('Convolutional architecture trained with EP on an Ising machine \n')
    f.write('   Parameters of the simulation \n ')
    f.write('\n')
    

    for key in args.__dict__:
        f.write(key)
        f.write(': ')
        f.write(str(args.__dict__[key]))
        f.write('\n')

    f.close()


def save_model_numpy(path, net):
    '''
    Save the parameters of the model as a dictionnary in a pickel file
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    with open(path + prefix + 'model_parameters.pickle', 'wb') as f:
            pickle.dump(net, f)

    return 0


def load_model_numpy(path):
    '''
    Save the parameters of the model as a dictionnary in a pickel file
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    with open(path + prefix + 'model_parameters.pickle', 'rb') as f:
            net = pickle.load(f)

    return net