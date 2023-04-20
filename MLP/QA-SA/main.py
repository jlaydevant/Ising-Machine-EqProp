import dimod
import neal
import matplotlib.pyplot as plt
import numpy as np
import argparse
from math import*
from Tools import*
from Network import*
from random import*
from tqdm import tqdm
import os

import torch

import dwave.inspector
from dwave.system import EmbeddingComposite, DWaveSampler, DWaveCliqueSampler, LazyFixedEmbeddingComposite
from simulated_sampler import SimulatedAnnealingSampler

if os.name != 'posix':
    prefix = '\\'
else:
    prefix = '/'

parser = argparse.ArgumentParser(description='Binary Equilibrium Propagation with D-Wave support')
#Architecture settings
parser.add_argument(
    '--dataset',
    type=str,
    default='mnist',
    help='Dataset we use for training (default=2d_class, others: digits)')
parser.add_argument(
    '--simulated',
    type=int,
    default=1,
    help='specify if we use simulated annealing (=1) or quantum annealing (=0) (default=0, else = 1)')
parser.add_argument(
    '--mode',
    type=str,
    default='ising',
    help='Which problem we submit to the QPU (default=qubo (0/1 spins), others: ising (-1/+1 variables: à priori no rescaling après))')
parser.add_argument(
    '--layersList',
    nargs='+',
    type=int,
    default=[784, 120, 40],
    help='List of layer sizes (default: 10)')
parser.add_argument(
    '--n_iter_free',
    type=int,
    default=10,
    help='Times to iterate for the QPU on a single data point for getting the minimal energy state of the free phase (default=1)')
parser.add_argument(
    '--n_iter_nudge',
    type=int,
    default=10,
    help='Times to iterate for the QPU on a single data point for getting the minimal energy state of the nudge phase (default=5)')
parser.add_argument(
    '--frac_anneal_nudge',
    type=float,
    default=0.25,
    help='fraction of system non-annealed (default=0.5, if <0.5: more annealing, if > 0.5, less annealing)')
parser.add_argument(
    '--N_data',
    type=int,
    default=1000,
    help='Number of data points for training (default=100)')
parser.add_argument(
    '--N_data_test',
    type=int,
    default=100,
    help='Number of data points for testing (default=10)')
parser.add_argument(
    '--beta',
    type=float,
    default=5,
    help='Beta - hyperparameter of EP (default=10)')
parser.add_argument(
    '--batch_size',
    type=int,
    default=1,
    help='Size of mini-batches we use (for training only)')
parser.add_argument(
    '--lrW0',
    type=float,
    default=0.01,
    help='Learning rate for weights - input-hidden  (default=0.01)')
parser.add_argument(
    '--lrW1',
    type=float,
    default=0.01,
    help='Learning rate for weights - hidden-output (default=0.01)')
parser.add_argument(
    '--lrB0',
    type=float,
    default=0.001,
    help='Learning rate for biases - hidden (default=0.001)')
parser.add_argument(
    '--lrB1',
    type=float,
    default=0.001,
    help='Learning rate for biases - output (default=0.001)')
parser.add_argument(
    '--epochs',
    type=int,
    default=50,
    help='Number of epochs (default=10)')
parser.add_argument(
    '--load_model',
    type=int,
    default=0,
    help='If we load the parameters from a previously trained model to continue the training (default=0, else = 1)')
parser.add_argument(
    '--gain_weight0',
    type=float,
    default=0.5,
    help='Gain for initialization of the weights - input-hidden (default=1)')
parser.add_argument(
    '--gain_weight1',
    type=float,
    default=0.25,
    help='Gain for initialization of the weights  - hidden-output (default=1)')
parser.add_argument(
    '--bias_lim',
    type=float,
    default=4.0,
    help='Max limit for the amplitude of the local biases applied to the qbits, either for free or nudge phase - (default=1)')
parser.add_argument(
    '--chain_strength',
    type=float,
    default=1.0,
    help='Value of the coupling in the chain of identical qbits - (default=4)')
parser.add_argument(
    '--auto_scale',
    type=int,
    default=0,
    help='Set auto_scale or not for the problems - (default=False)')
args = parser.parse_args()

if args.auto_scale == 0:
    args.auto_scale = False
else:
    args.auto_scale = True

with torch.no_grad():
    ## SAMPLERs
    simu_sampler = neal.SimulatedAnnealingSampler()
    exact_sampler = dimod.ExactSolver()

    if args.simulated == 1:
        qpu_sampler = SimulatedAnnealingSampler()
    else:
        qpu_sampler = LazyFixedEmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}, auto_scale = args.auto_scale))

    ## Files saving: create a folder for the simulation and save simulation's parameters
    BASE_PATH = createPath(args, simu = '')
    dataframe = initDataframe(BASE_PATH)
    print(BASE_PATH)

    ## Generate DATA
    if args.dataset == "digits":
        train_loader, test_loader = generate_digits(args)
    elif args.dataset == "mnist":
        train_loader, test_loader, dataset = generate_mnist(args)


    ## Create the network
    if args.load_model == 0:
        saveHyperparameters(BASE_PATH, args, simu = 'comparaison-ExactSolver-QPUSolver')
        net = Network(args)
    else:
        net = load_model_numpy(BASE_PATH)

    ## Monitor loss and prediction error
    exact_loss_tab, exact_falsePred_tab = [], []
    exact_loss_test_tab, exact_falsePred_test_tab = [], []

    qpu_loss_tab, qpu_falsePred_tab = [], []
    qpu_loss_test_tab, qpu_falsePred_test_tab = [], []

    for epoch in tqdm(range(args.epochs)):
        # Train the network
        exact_loss, exact_falsePred, qpu_loss, qpu_falsePred = train(net, args, train_loader, simu_sampler, exact_sampler, qpu_sampler)
        exact_loss_tab.append(exact_loss)
        exact_falsePred_tab.append(exact_falsePred)
        qpu_loss_tab.append(qpu_loss)
        qpu_falsePred_tab.append(qpu_falsePred)

        # Test the network
        exact_loss, exact_falsePred, qpu_loss, qpu_falsePred = test(net, args, test_loader, simu_sampler, exact_sampler, qpu_sampler)
        exact_loss_test_tab.append(exact_loss)
        exact_falsePred_test_tab.append(exact_falsePred)
        qpu_loss_test_tab.append(qpu_loss)
        qpu_falsePred_test_tab.append(qpu_falsePred)

        # Store error and loss at each epoch
        dataframe = updateDataframe(BASE_PATH, dataframe, np.array(exact_falsePred_tab)[-1]/len(train_loader.dataset)*100, np.array(exact_falsePred_test_tab)[-1]/len(test_loader.dataset)*100, np.array(qpu_falsePred_tab)[-1]/len(train_loader.dataset)*100, np.array(qpu_falsePred_test_tab)[-1]/len(test_loader.dataset)*100, exact_loss_tab, exact_loss_test_tab, qpu_loss_tab, qpu_loss_test_tab)

        save_model_numpy(BASE_PATH, net)