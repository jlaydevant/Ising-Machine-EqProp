import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
import torch

class Network(nn.Module):

    def __init__(self, args):

        super(Network, self).__init__()

        self.sign_beta = 1
        with torch.no_grad():
            #init the parameters of the network: weights & biases - with the method provided by the parser
            N_inputs, N_hidden, N_output =  args.layersList[0], args.layersList[1], args.layersList[2]
            if args.init == 'random_normal':
                self.weights_0 = np.random.randn(N_inputs, N_hidden)
                self.weights_1 = np.random.randn(N_hidden, N_output)
            elif args.init == 'random_uniform':
                self.weights_0 = 2*(np.random.rand(N_inputs, N_hidden)-0.5)
                self.weights_1 = 2*(np.random.rand(N_hidden, N_output)-0.5)
            elif args.init == 'kaiming_normal':
                self.weights_0 = np.random.randn(N_inputs, N_hidden)*math.sqrt(3/N_inputs)
                self.weights_1 = np.random.randn(N_hidden, N_output)*math.sqrt(3/N_hidden)
            elif args.init == 'kaiming_uniform':
                self.weights_0 = 2*(np.random.rand(N_inputs, N_hidden)-0.5)*math.sqrt(1/N_inputs)
                self.weights_1 = 2*(np.random.rand(N_hidden, N_output)-0.5)*math.sqrt(1/N_hidden)
            elif args.init == 'glorot_normal':
                self.weights_0 = np.random.randn(N_inputs, N_hidden)*math.sqrt(2/(N_inputs+N_hidden))
                self.weights_1 = np.random.randn(N_hidden, N_output)*math.sqrt(2/(N_hidden+N_output))
            elif args.init == 'glorot_uniform':
                self.weights_0 = 2*(np.random.rand(N_inputs, N_hidden)-0.5)*math.sqrt(6/(N_inputs+N_hidden))
                self.weights_1 = 2*(np.random.rand(N_hidden, N_output)-0.5)*math.sqrt(6/(N_hidden+N_output))

            self.weights_0 = args.gain_weight0 * self.weights_0
            self.weights_1 = args.gain_weight1 * self.weights_1

            self.bias_0 = args.gain_bias0*np.random.randn(N_hidden)
            self.bias_1 = args.gain_bias1*np.random.randn(N_output)
            # self.bias_0 = -0.01*np.ones(N_hidden)
            # self.bias_1 = -0.01*np.ones(N_output)


            self.confusion_matrix_train = np.zeros((10,10))
            self.confusion_matrix_test = np.zeros((10,10))


    def computeLossError(self, seq, target, args, stage = 'training'):
        '''
        compute the loss and the error from s and seq (for any kind of sampler)
        '''
        with torch.no_grad():
            if args.dataset == "2d_class":
                expand_output = int(args.layersList[2]/2)
            elif args.dataset == "digits" or args.dataset == "mnist":
                expand_output = int(args.layersList[2]/10)

            #print("---------------------------")
            #print(target)

            #sanity check
            assert seq[1].shape == target.shape

            loss = (((target-seq[1])**2).sum()/2).item()

            pred_ave   = np.stack([item.sum(1) for item in np.split(seq[1], int(args.layersList[2]/expand_output), axis = 1)], 1)/expand_output
            target_red = np.stack([item.sum(1) for item in np.split(target, int(args.layersList[2]/expand_output), axis = 1)], 1)/expand_output

            assert pred_ave.shape == target_red.shape
            #print(pred_ave)
            #print(target_red)
            #print(pred_ave.shape)
            #print(target_red.shape)
            #print("target_argmax = " + str(np.argmax(target_red, axis = 1)))
            #print("pred_argmax = " + str(np.argmax(pred_ave, axis = 1)))
            #checker ici si on a bien des arrays de dimension 2 même si on n'a qu'une image à la fois !!
            pred = ((np.argmax(target_red, axis = 1) == np.argmax(pred_ave, axis = 1))*1).sum()

            if stage == 'training':
                for k in range(len(target_red)):
                    self.confusion_matrix_train[np.argmax(target_red, axis = 1)[k],np.argmax(pred_ave, axis = 1)[k]] += 1

            elif stage == 'testing':
                for k in range(len(np.argmax(target_red, axis = 1))):
                    self.confusion_matrix_test[np.argmax(target_red, axis = 1)[k],np.argmax(pred_ave, axis = 1)[k]] += 1

        return loss, pred


    def computeGrads(self, data, s, seq, args):
        with torch.no_grad():
            coef = self.sign_beta*args.beta*args.batch_size
            #print(coef)
            gradsW, gradsB = [], []

            gradsW.append(-(np.matmul(s[0].T, s[1]) - np.matmul(seq[0].T, seq[1])) /coef)
            gradsW.append(-(np.matmul(data.numpy().T, s[0]) - np.matmul(data.numpy().T, seq[0])) /coef)

            gradsB.append(-(s[1] - seq[1]).sum(0) /coef)
            gradsB.append(-(s[0] - seq[0]).sum(0) /coef)


            return gradsW, gradsB


    def updateParams(self, data, s, seq, args):
        with torch.no_grad():
            ## Compute gradients and update weights from simulated sampling
            gradsW, gradsB = self.computeGrads(data, s, seq, args)

            #print("===========================================")
            #print(gradsW)
            #print(gradsB)
            #le gradient moyen ne veut pas dire grand chose, regarder en même temps l'écart-type, puisque bcp de gradients peuvent être à 0, avec des gradients non-nuls élevés, donc la moyenne n'est pas représentative !
            #print("Gradient B couche 1: " + str(gradsB[0]))
            #print("Gradient W moyen couche 1: " + str(np.mean(np.abs(gradsW[0]))) + " - Grad max = " + str(np.max(np.abs(gradsW[0]))) + " - Grad std = " + str(np.std(np.abs(gradsW[0]))))
            #print("Gradient W moyen couche 0: " + str(np.mean(np.abs(gradsW[1]))) + " - Grad max = " + str(np.max(np.abs(gradsW[1]))) + " - Grad std = " + str(np.std(np.abs(gradsW[1]))))
            #print("Gradient B moyen couche 1: " + str(np.mean(np.abs(gradsB[0]))) + " - Grad max = " + str(np.max(np.abs(gradsB[0]))) + " - Grad std = " + str(np.std(np.abs(gradsB[0]))))
            #print("Gradient B moyen couche 0: " + str(np.mean(np.abs(gradsB[1]))) + " - Grad max = " + str(np.max(np.abs(gradsB[1]))) + " - Grad std = " + str(np.std(np.abs(gradsB[1]))))

            #weights
            assert self.weights_1.shape == gradsW[0].shape
            self.weights_1 += args.lrW0 * gradsW[0]
            #self.weights_1 = self.weights_1.clip(-1,1)

            assert self.weights_0.shape == gradsW[1].shape
            self.weights_0 += args.lrW1 * gradsW[1]
            #self.weights_0 = self.weights_0.clip(-1,1)

            #biases
            assert self.bias_1.shape == gradsB[0].shape
            self.bias_1 += args.lrB0 * gradsB[0]
            #self.bias_1 = self.bias_1.clip(-1,1)

            assert self.bias_0.shape == gradsB[1].shape
            self.bias_0 += args.lrB1 * gradsB[1]
            #self.bias_0 = self.bias_0.clip(-1,1)

            del gradsW, gradsB

