import os
import argparse
import torchvision.datasets as datasets
import torchvision
import torch.nn.functional as F
import datetime
import numpy as np
import platform
import time
from tqdm import tqdm

from Tools import *
from Network import *

import dwave.inspector
from dwave.system import FixedEmbeddingComposite, EmbeddingComposite, DWaveSampler, DWaveCliqueSampler
import neal, dimod

if os.name != 'posix':
    prefix = '\\'
else:
    prefix = '/'

parser = argparse.ArgumentParser(description='Training a convolutional neural network on an Ising machine with Equilibrium Propagation')
#Architecture settings
parser.add_argument(
    '--layersList',
    nargs='+',
    type=int,
    default=[4],
    help='Number of output neurons (default: 4)')
parser.add_argument(
    '--expand_output',
    type=int,
    default=2,
    help='Number by how much we enlarge the output layer compared to a standard 1-to-1 correspondance (default=2)')
parser.add_argument(
    '--convList',
    nargs='+',
    type=int,
    default=[4, 1],
    help='List of channels for the convolution - first for the convolutional layer, second for the input data (default = [4, 1])')
parser.add_argument(
    '--padding',
    type=int,
    default=0,
    metavar='P',
    help='Padding for the convolution (default: 0)')
parser.add_argument(
    '--kernelSize',
    type=int,
    default=2,
    metavar='KSize',
    help='Kernel size for convolution (default: 2)')
parser.add_argument(
    '--Fpool',
    type=int,
    default=2,
    metavar='Fp',
    help='Average Pooling filter size (default: 2)')
parser.add_argument(
    '--pool_coef',
    type=float,
    default=0.25,
    help='coefficient used for doing the averaged pooling operation (default: 0.25)')

#EqProp settings
parser.add_argument(
    '--mode',
    type=str,
    default='ising',
    help='Type of Ising problem submitted to the QPU (default=ising (-1/1 spins), others: qubo (0/1 variables))')
parser.add_argument(
    '--beta',
    type=float,
    default=5,
    help='nudging parameter (default: 5)')
parser.add_argument(
    '--n_iter_free',
    type=int,
    default=10,
    help='Number of iterations for the free phase on a single data point (default=10)')
parser.add_argument(
    '--n_iter_nudge',
    type=int,
    default=10,
    help='Number of iterations for the nudge phase on a single data point  (default=10)')
parser.add_argument(
    '--frac_anneal_nudge',
    type=float,
    default=0.25,
    help='Fraction of system non-annealed (default=0.25, if <0.25: more annealing, if > 0.25, less annealing)')
#Training settings
parser.add_argument(
    '--dataset',
    type=str,
    default='patterns',
    help='dataset used to train the network (default=patterns)')
parser.add_argument(
    '--lrWeightsFC',
    nargs='+',
    type=float,
    default=[0.1],
    help='learning rates for bias')
parser.add_argument(
    '--lrWeightsCONV',
    nargs='+',
    type=float,
    default=[0.1],
    help='learning rates for bias')
parser.add_argument(
    '--lrBiasFC',
    nargs='+',
    type=float,
    default=[0],
    help='learning rates for bias')
parser.add_argument(
    '--lrBiasCONV',
    nargs='+',
    type=float,
    default=[0],
    help='learning rates for bias')
parser.add_argument(
    '--batchSize',
    type=int,
    default=1,
    help='Batch size (default=10)')
parser.add_argument(
    '--epochs',
    type=int,
    default=20,
    metavar='N',
help='number of epochs to train (default: 1)')
parser.add_argument(
    '--chain_strength',
    type=float,
    default=2.0,
    help='Value of the coupling in the chain of identical spins - (default=2)')
parser.add_argument(
    '--auto_scale',
    type=int,
    default=0,
    help='Set auto_scale for the problems - (default=False)')
args = parser.parse_args()

if args.auto_scale == 0:
    args.auto_scale = False
else:
    args.auto_scale = True


with torch.no_grad():
    ## SAMPLER
    if args.simulated == 1:
        #Simulated annealing sampler
        sampler = SimulatedAnnealingSampler()
    else:
        # D-Wave solver: DW2000 with chimera topology as the conv architecture suits better this topology
        emb_list = {0: [560], 1: [561], 2: [562], 3: [563], 4: [444], 5: [445], 6: [446], 7: [447], 8: [460], 9: [461], 10: [462], 11: [463], 12: [592], 13: [593], 14: [594], 15: [595], 16: [564], 17: [565], 18: [566], 19: [567], 20: [440], 21: [441], 22: [442], 23: [443], 24: [456], 25: [457], 26: [458], 27: [459], 28: [596], 29: [597], 30: [598], 31: [599], 32: [568, 572, 580, 584, 588], 33: [569, 573, 581, 585, 589], 34: [570, 574, 582, 586, 590], 35: [571, 575, 583, 587, 591], 36: [576], 37: [577], 38: [578], 39: [579]}
        sampler = FixedEmbeddingComposite( DWaveSampler(solver={'topology__type': 'chimera'}, auto_scale = args.auto_scale)sampler, embedding = emb_list)

    ## Files saving: create a folder for the simulation and save simulation's parameters
    BASE_PATH = createPath(args)
    dataframe = initDataframe(BASE_PATH)

    ## Generate DATA
    train_loader, test_loader = generate_patterns(args)

    ## Create the network
    saveHyperparameters(BASE_PATH, args)
    net = Network(args)

    ## Monitor loss and prediction error
    qpu_loss_tab, qpu_falsePred_tab = [], []
    qpu_loss_test_tab, qpu_falsePred_test_tab = [], []

    for epoch in tqdm(range(args.epochs)):
        # Train the network
        qpu_loss, qpu_falsePred = train(net, args, train_loader, sampler)
        qpu_loss_tab.append(qpu_loss)
        qpu_falsePred_tab.append(qpu_falsePred)

        # Store error and loss at each epoch
        dataframe = updateDataframe(BASE_PATH, dataframe, np.array(qpu_falsePred_tab)[-1]/len(train_loader.dataset)*100, qpu_loss_tab[-1])
        save_model_numpy(BASE_PATH, net)
        plot_functions(net, BASE_PATH, prefix, epoch)

