# coding: utf-8
#Main for the simulation
import os
import argparse
import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import pickle
import datetime
import numpy as np
import platform
import time
import sklearn
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split


from tqdm import tqdm

from Tools import *
from Network import *
from plotFunction import*

parser = argparse.ArgumentParser(description='Train a MLP with Equilibrium Propagation - real-valued weights & binary neurons')
parser.add_argument(
    '--device',
    type=int,
    default=-1,
    help='GPU name to use cuda')
parser.add_argument(
    '--dataset',
    type=str,
    default='mnist',
    help='Dataset we use for training (default=mnist, others: digits)')
parser.add_argument(
    '--N_data',
    type=int,
    default=1000,
    metavar='N',
help='number of training images (default: 1000)')
parser.add_argument(
    '--N_data_test',
    type=int,
    default=100,
    metavar='N',
help='number of training images (default: 100)')
parser.add_argument(
    '--epochs',
    type=int,
    default=0,
    metavar='N',
help='number of epochs to train (default: 1)')
parser.add_argument(
    '--batchSize',
    type=int,
    default=1,
    help='Batch size (default=10)')
parser.add_argument(
    '--test_batchSize',
    type=int,
    default=512,
    help='Testing Batch size (default=512)')
parser.add_argument(
    '--T',
    type=int,
    default=20,
    metavar='T',
    help='number of time steps in the free phase (default: 50)')
parser.add_argument(
    '--Kmax',
    type=int,
    default=10,
    metavar='Kmax',
    help='number of time steps in the backward pass (default: 10)')
parser.add_argument(
    '--beta',
    type=float,
    default=2,
    help='nudging parameter (default: 1)')
parser.add_argument(
    '--gamma_neur',
    type=float,
    default=5e-1,
    help='gamma to filter out pre-activations of neurons for relaxation')
parser.add_argument(
    '--clamped',
    type=int,
    default=1,
    help='Clamped neurons or not: crossed input are clamped to avoid divergence  (default: True)')
parser.add_argument(
    '--activationFun',
    type=str,
    default="heavyside",
    help='Binary activation function (: sign, heavyside)')
parser.add_argument(
    '--rho_threshold',
    type=float,
    default=0.5,
    help='threshold/offset of the activation function! 0.5 mean rho(x-0.5), 0 for rho(x)')
parser.add_argument(
    '--layersList',
    nargs='+',
    type=int,
    default=[784, 120, 40],
    help='List of layers in the model')
parser.add_argument(
    '--lrW',
    nargs='+',
    type=float,
    default=[1e-2, 1e-2],
    help="learning rates for each layer's weigths")
parser.add_argument(
    '--lrB',
    nargs='+',
    type=float,
    default=[1e-3, 1e-3],
    help="learning rates for each layer's biases")
parser.add_argument(
    '--expand_output',
    type=int,
    default=4,
    help='quantity by how much we increase the output layer (default=10)')
parser.add_argument(
    '--weightClip',
    type=float,
    default=1,
    help='Limit to which clip the weights after SGD update (default=100: no weight clipping)')
parser.add_argument(
    '--biasClip',
    type=float,
    default=2,
    help='Limit to which clip the biases after SGD update (default=100: no bias clipping)')


args = parser.parse_args()


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class ReshapeTransformTarget:
    def __init__(self, number_classes, activFun):
        self.number_classes = number_classes
        self.activFun = activFun

    def __call__(self, target):
        target=torch.tensor(target).unsqueeze(0).unsqueeze(1)
        target_onehot = torch.zeros((1,self.number_classes))
        return target_onehot.scatter_(1, target.long(), 1).repeat_interleave(args.expand_output).squeeze(0)


elif args.dataset == "digits":
    train_loader, test_loader = generate_digits(args)

elif args.dataset == "mnist":
    train_loader, test_loader, dataset = generate_mnist(args)

def rho(x):
    return (((torch.sign(x)+1)/2)*(x != 0).float() + (x == 0).float()).float()
def rhop(x):
    #we use this convention as the x is pre-centered by args.rho_threshold before, so we say it is one if -0.5<x<0.5
    return ((x >= -0.5) & (x <= 0.5)).float()


if __name__ == '__main__':
    with torch.no_grad():
        args.layersList.reverse()

        net = Network(args)
        if net.cuda is True:
            net = net.to(net.device)

        if os.name != 'posix':
            prefix = '\\'
        else:
            prefix = '/'

        BASE_PATH, name = createPath(args)
        saveHyperparameters(args, net, BASE_PATH)
        DATAFRAME = initDataframe(BASE_PATH, args, net)

        for epoch in tqdm(range(args.epochs)):
            ave_train_error, single_train_error, train_loss = train_bin(net, args, train_loader, epoch)
            ave_test_error, single_test_error, test_loss = test_bin(net, args, test_loader)

            DATAFRAME = updateDataframe(BASE_PATH, args, DATAFRAME, net, ave_train_error.cpu().item(), ave_test_error.cpu().item(), single_train_error.cpu().item(), single_test_error.cpu().item(), train_loss.cpu().item(), test_loss.cpu().item())
            torch.save(net.state_dict(), BASE_PATH + prefix + 'model_state_dict.pt')


