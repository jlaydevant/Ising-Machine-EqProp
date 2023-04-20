from scipy import*
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, args):

        super(Network, self).__init__()

        self.beta = args.beta
        self.batchSize = args.batchSize

        self.kernelSize = args.kernelSize
        self.Fpool = args.Fpool
        self.convList = args.convList
        self.layersList = args.layersList
        
        self.n_cp = len(args.convList) - 1
        self.n_classifier = len(args.layersList)
        
        P = args.padding

        self.P = args.padding

        self.conv = nn.ModuleList([])
        self.fc = nn.ModuleList([])  
        self.pool = nn.ModuleList([])
        self.Fpool = args.Fpool

        input_size = 3
        
        self.size_convpool_tab = [input_size] 
        self.size_conv_tab = [input_size]

        for i in range(self.n_cp):
            self.conv.append(nn.Conv2d(args.convList[i + 1], args.convList[i], args.kernelSize, padding = P, stride = 1))
            self.pool.append(nn.Conv2d(args.convList[i], args.convList[i], kernel_size = (args.Fpool, args.Fpool), stride = (args.Fpool, args.Fpool), bias = False, groups = args.convList[i]))

            torch.nn.init.zeros_(self.conv[-1].bias)
            self.pool[-1].weight.data = 1/(args.Fpool)*torch.ones(self.pool[-1].weight.size())
                        
            self.size_conv_tab.append(int((self.size_convpool_tab[i] + 2*P - args.kernelSize)/1 + 1 ))
            
            self.size_convpool_tab.append(int((self.size_conv_tab[-1]-args.Fpool+2*0)/args.Fpool)+1)

        self.size_convpool_tab.reverse()
        self.size_conv_tab.reverse()
        
        
        self.nconv = len(self.size_convpool_tab) - 1

        self.layersList.append(args.convList[0]*self.size_convpool_tab[0]**2)

        self.nc = len(self.layersList) - 1
        
        for i in range(self.n_classifier):
            self.fc.append(nn.Linear(self.layersList[i + 1], self.layersList[i]))    
            torch.nn.init.zeros_(self.fc[-1].bias)
            
            
        self.index_before_pool = [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
        self.index_after_pool  = [32,33,34,35]
        self.index_fc          = [36,37,38,39]

    
    def sample_to_s(self, args, data, sample):
        '''
        Function that takes as an input the sample returned by the qpu and reshape 
        it as the list s we usually use for computing the gradient
        '''
        s = self.initHidden(args, data)
        
        comp = 16 #the first 0:15 spins are related to input data - we already have their value as it is the value of the input data
        
        for img_idx in range(data.size(0)):
            #neurons after convolution
            for k in range(args.convList[0]):
                for i in range(self.size_conv_tab[0]):
                    for j in range(self.size_conv_tab[0]):
                        s[0][img_idx][k][i][j] = sample[img_idx][comp]
                        comp += 1
            #neurons after averaged pooling
            for k in range(args.convList[0]):
                s[1][img_idx][k][0][0] = sample[img_idx][comp]
                comp += 1

            #neurons after fully connected layer (output neurons)
            for k in range(args.layersList[1]):
                s[2][img_idx][k] = sample[img_idx][comp]
                comp += 1

            assert len(sample[img_idx]) == comp
        
        return s
        
        
    def computeLossAcc(self, seq, target, args, stage = 'training'):
        '''
        compute the loss and the error from s and seq (for any kind of sampler)
        '''
        with torch.no_grad():
            expand_output = int(args.layersList[0]/2)
            
            assert seq[-1].shape == target.shape
            loss = (((target-seq[-1])**2).sum()/2).item()
            
            pred_ave   = np.stack([item.sum(1) for item in np.split(seq[-1], int(args.layersList[0]/expand_output), axis = 1)], 1)/expand_output
            target_red = np.stack([item.sum(1) for item in np.split(target, int(args.layersList[0]/expand_output), axis = 1)], 1)/expand_output

        
            assert pred_ave.shape == target_red.shape
            pred = ((np.argmax(target_red, axis = 1) == np.argmax(pred_ave, axis = 1))*1).sum()

        return loss, pred
    
    
    def computeGradients(self, args, s, seq, data):
        '''
        We compute the gradients from the two state sampled on the QPU
        The way we embed the architecture on the chip, we don't need to compute the unpool operation and juste need to compute the convolution between the input and the convolution results! Also, we just need to compute the matrix product between the result of the average pooling and the output layer!
        '''

        batch_size = s[0].size(0)
        coef = 1./float((self.beta*batch_size))
               
        #CLASSIFIER     
        gradfc = -coef*(torch.mm(torch.transpose(s[2].view(seq[0].size(0),-1), 0, 1), s[1].view(seq[0].size(0),-1)) - torch.mm(torch.transpose(seq[2].view(seq[0].size(0),-1), 0, 1), seq[1].view(seq[0].size(0),-1)))         
        gradfc_bias = -coef*(s[2] - seq[2]).sum(0) 
                                                            
        #CONVOLUTIONAL
        gradconv = -coef*(F.conv2d(data, weight = s[0].view(4,1,2,2), padding = 0, stride = 1)-F.conv2d(data, weight = seq[0].view(4,1,2,2), padding = 0, stride = 1)).view(self.conv[0].weight.size())
        gradconv_bias = -coef*(s[0]-seq[0]).sum(1)
            
        return gradfc, gradfc_bias, gradconv, gradconv_bias

 
    def updateParams(self, data, s, seq, args):
        
        with torch.no_grad():
            gradfc, gradfc_bias, gradconv, gradconv_bias = self.computeGradients(args, s, seq, data)
         
            #update weights
            assert self.fc[0].weight.size() == gradfc.size()  
            self.fc[0].weight.data += args.lrWeightsFC[0]*gradfc
            self.fc[0].weight.data = self.fc[0].weight.data.clip(-1,1)

            #update bias
            assert self.fc[0].bias.size() == gradfc_bias.size()
            self.fc[0].bias += args.lrBiasFC[0]*gradfc_bias

            #update weights
            assert self.conv[0].weight.size() == gradconv.size()
            self.conv[0].weight += args.lrWeightsCONV[0]*gradconv
            self.conv[0].weight.data = self.conv[0].weight.data.clip(-1,1)

        return 0

    
    







        
        
