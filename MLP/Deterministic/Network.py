from scipy import*
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

try:
    from main import rho, rhop1, rhop2
except:
    from main import rho, rhop

class Network(nn.Module):
    ''' Define the network studied
    '''
    def __init__(self, args):

        super(Network, self).__init__()

        self.T = args.T
        self.Kmax = args.Kmax
        self.beta = torch.tensor(args.beta)
        self.gamma_neur = args.gamma_neur
        self.clamped = args.clamped
        self.batchSize = args.batchSize

        self.neuronMin = 0
        self.neuronMax = 1

        self.activationFun = args.activationFun
        self.epoch = 0

        self.W = nn.ModuleList(None)
        with torch.no_grad():
            for i in range(len(args.layersList)-1):
                self.W.extend([nn.Linear(args.layersList[i+1], args.layersList[i], bias = True)])

        if args.device >= 0 and torch.cuda.is_available():
            self.device = torch.device("cuda:"+str(args.device))
            self.cuda = True
        else:
            self.device = torch.device("cpu")
            self.cuda = False


    def getBinState(self, states, args):
        '''
        Return the binary states from pre-activations stored in 'states'
        '''
        bin_states = states.copy()

        for layer in range(len(states)-1):
            bin_states[layer] = rho(states[layer] - args.rho_threshold)

        bin_states[-1] = states[-1]

        return bin_states


    def stepper(self, args, s, seq = None, target = None, beta = 0):
        '''
        EP based model - prototypical settings, ie not energy based dynamics!
        with moving average filter for pre-activations
        '''
        pre_act = s.copy()
        bin_states = self.getBinState(s, args)

        #computing pre-activation for every layer weights + bias
        pre_act[0] = self.W[0](bin_states[1])
        pre_act[0] = rhop(s[0] - args.rho_threshold)*pre_act[0]
        if beta != 0:
            pre_act[0] = pre_act[0] + beta*(target-s[0])

        for layer in range(1, len(s)-1):
            #previous layer contribution: weights + bias
            pre_act[layer] =  self.W[layer](bin_states[layer+1])
            # next layer contribution
            pre_act[layer] += torch.mm(bin_states[layer-1], self.W[layer-1].weight)
            #multiply with Indicatrice(pre-activation)
            pre_act[layer] = rhop(s[layer] - args.rho_threshold)*pre_act[layer]

        #updating each accumulated pre-activation
        for layer in range(len(s)-1):
            #moving average filter for the pre_activations
            s[layer] =  (1-self.gamma_neur)*s[layer] + self.gamma_neur*pre_act[layer]

            #clamping on pre-activations
            s[layer] = s[layer].clamp(self.neuronMin, self.neuronMax)


        return s


    def forward(self, args, s, seq = None,  beta = 0, target = None, tracking = False):
        '''
        Relaxation function
        '''
        T, Kmax = self.T, self.Kmax
        n_track = 10
        h, y = [[] for k in range(n_track)], [[] for k in range(n_track)]

        with torch.no_grad():
            if beta == 0:
                # Free phase
                for t in range(T):
                    s = self.stepper(args, s)

                    if tracking:
                        for k in range(n_track):
                            y[k].append(s[0][k][2*k].item())
                            h[k].append(s[1][k][2*k].item())
            else:
                # Nudged phase
                for t in range(Kmax):
                    s = self.stepper(args,s, target = target, beta = beta, seq = seq)

                    if tracking:
                        for k in range(n_track):
                            y[k].append(s[0][k][2*k].item())
                            h[k].append(s[1][k][2*k].item())
        if tracking:
            return s, y, h
        else:
            return s


    def computeGradients(self, args, s, seq, method = None):
        '''
        Compute EQ gradient to update the synaptic weight
        '''
        batch_size = s[0].size(0)

        coef = 1/(self.beta*batch_size)
        gradW, gradBias, gradAlpha = [], [], []

        with torch.no_grad():
            bin_states_0, bin_states_1 = self.getBinState(seq, args), self.getBinState(s, args)

            for layer in range(len(s)-1):
                gradW.append( coef * (torch.mm(torch.transpose(bin_states_1[layer], 0, 1), bin_states_1[layer+1]) - torch.mm(torch.transpose(bin_states_0[layer], 0, 1), bin_states_0[layer+1])))
                gradBias.append( coef * (bin_states_1[layer] -  bin_states_0[layer]).sum(0))

        return gradW, gradBias


    def updateWeight(self, epoch, s, seq, args):
        '''
        Update weights with SGD
        '''

        gradW, gradBias = self.computeGradients(args, s, seq)

        with torch.no_grad():
            for i in range(len(s)-1):
                #update weights: SGD + weight clipping
                self.W[i].weight += args.lrW[i] * gradW[i]
                self.W[i].weight.data = self.W[i].weight.clamp(-args.weightClip, +args.weightClip).to(self.device)

                #update biases
                self.W[i].bias += args.lrB[i] * gradBias[i]
                self.W[i].bias.clamp_(-args.biasClip, +args.biasClip)

        return 0


    def initHidden(self, args, data, testing = False):
        '''
        Init the state of the network
        State if a dict, each layer is state["S_layer"]
        Xdata is the the last element of the dict
        '''
        state = []
        size = data.size(0)

        for layer in range(len(args.layersList)-1):
            state.append(torch.zeros(size, args.layersList[layer], requires_grad = False))

        state.append(data.float())

        return state




