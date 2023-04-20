# Training an Ising Machine with Equilibrium Propagation
This repository contains the code to reproduce the results of the paper "Training an Ising Machine with Equilibrium Propagation".

The project contains the following folder:

+ `MLP/QA-SA`: contain the code to train the fully-connected neural network with an Ising machine (the D-Wave Ising machine or with Simulated annealing - /!\ the D-Wave Ising machine requires the user to have acces to the machine)

+ `MLP/Deterministic`: contain the code to train the fully-connected neural network with a deterministic dynamics (with binary neurons and real-value weights)

+ `Conv/QA-SA`: contain the code to train the small convolutional neural network with an Ising machine (the D-Wave Ising machine or with Simulated annealing - /!\ the D-Wave Ising machine requires the user to have acces to the machine)


# Training the fully-connected architecture with an Ising machine (D-Wave, Simulated annealing)
The code for training the convolutional architecture can be found in the folder ```./MLP/QA-SA```.


## Details about main.py

The parser accepts the following arguments:

+ Hardware settings:

|Arguments|Description|Example|
|-------|------|------|
|`simulated`| 0 if the D-Wave Ising machine is used (need access to the D-Wave Ising machine) - 1 if the Simulated annealing sampler is used| `--simulated 0`|

+ Architecture settings:

|Arguments|Description|Example|
|-------|------|------|
|`layersList`|Number of neurons in the output layer| `--layerList 784 120 40`|
|`expand_output`|How much we duplicate the output neurons (required for nudging a binary system)| `--expand_output 2`|

+ EqProp settings:

|Arguments|Description|Example|
|-------|------|------|
|`mode`|Type of Ising problem to be submitted to the Ising machine - ising= {+/- 1 spins}, qubo: {0/1 spins}| `--mode ising`|
|`beta`|Nudging parameter| `-beta 5`|
|`n_iter_free`|Number of iterations for the free phase on a single data point| `--n_iter_free 10`|
|`n_iter_nudge`|Number of iterations for the nudge phase on a single data point| `--n_iter_nudge 10`|
|`frac_anneal_nudge`|Fraction of system non-annealed during the second phase| `--frac_anneal_nudge 0.25`|

+ Training settings:

|Arguments|Description|Example|
|-------|------|------|
|`dataset`|Dataset used to train the network| `--dataset mnist`|
|`N_data`|Number of training images (for MNIST dataset)|`--N_data mnist`|
|`N_data_test`|Number of testing images (for MNIST dataset)|`--N_data_test mnist`|
|`lrW0`|Learning rate for the weights of the fully-connected layer between the input and the hidden layer| `--lrW0 0.01`|
|`gain_weight0`|Multiplicative gain for initializing the weights of the fully-connected layer between the input and the hidden layer| `--gain_weight0 0.5`|
|`lrW1`|Learning rate for the weights of the fully-connected layer between the hidden and the output layer| `--lrW1 0.01`|
|`gain_weight1`|Multiplicative gain for initializing the weights of the fully-connected layer between the hidden and output layer| `--gain_weight1 0.25`|
|`lrB0`|Learning rate for the biases of the hidden layer| `--lrB0 0.001`|
|`lrB1`|Learning rate for the biases of the output layer| `--lrB1 0.001`|
|`bias_lim`|Maximal absolute value for the biases| `--bias_lim 4`|
|`batchSize`|Batch size (train and test)| `--batchSize 1`|
|`epochs`|Number of epochs| `--epochs 20`|
|`chain_strength`|Value of the coupling in the chain of identical spins| `--chain_strength 1`|
|`auto_scale`|Enable (1) or disable (0) the auto-scale  feature| `--auto_scale 0`|
|`load_model`|Create a new model (0) or load (>0) a model already or partially trained to continue the training | `--load-model 0`|


## Details about `Network.py`:
Each file contains a class of a network. Each class inherits from a nn.Module.

Each network has the following built-in functions:

+ `init`: specifies the parameter and the architecture of the network.

+ `computeLossAcc`: compute the loss and if the network predict the good class for the input (accuracy) 

+ `computeGrads`: compute the instantaneous gradient for each parameter

+ `updateParams`: update all parameters in the network according to the gradient computed in computeGradient - vanilla SGD


    
## Details about `Tools.py`:
Each file contains the same functions but adapted for each architecture.

+ `generate_digits`: generate the digits dataset from sklearn.

+ `generate_mnist`: generate the MNIST/X dataset given the number of training and testing images specified in the parser.

+ `createBQM`: translates the neural network architecture into a Ising problem (ie convert the weights into couplings between nodes and biases into individual bias fields)

+ `train`: trains the network over the whole training dataset once. Run the free and nudge phase and update the weights. Track the number of changes in the weights.

+ `test`: tests the network over the whole testing dataset once.

+ `initDataframe`: inits the dataframe where we store the training data.

+ `updateDataframe`: updates the same dataframe after each epoch: update train error, test error, number of changes of weights for each layer.

+ `createPath`: creates the path and the folder where the training data will be stored. Copy the plotFunction.py file in the same folder. Create a folder specific to the training 'S-X' in this general folder.

+ `saveHyperparameters`: creates a .txt file with all hyperparameters in the parser in the folder 'S-X'.


## Commands to be run in the terminal to reproduce the results of the paper:
    ```
    python MLP/QA-SA/main.py --load_model 0 --simulated 1 --dataset mnist -- N_data 1000 --N_data_test 100 --layersList 784 120 40 --expand_output 4 --mode ising --beta 5 --n_iter_free 10 --n_iter_nudge 10 --frac_anneal_nudge 0.25 --lrW0 0.01 --gain_weight0 0.5 --lrW1 0.01 --gain_weight0 0.25 --lrB0 0.001 --lrB1 0.001 --batchSize 1 --epochs 50 --chain_strenght 1 --auto_scale 0
    ```


# Training the fully-connected architecture with a deterministic dynamics 
The code for training the convolutional architecture can be found in the folder ```./MLP/Deterministic```.

## Details about main.py

The parser accepts the following arguments:

+ Hardware settings:

|Arguments|Description|Example|
|-------|------|------|
|`device`| ID (0, 1, ...) of the GPU if provided - -1 for CPU| `--device 0`|

+ Architecture settings:

|Arguments|Description|Example|
|-------|------|------|
|`layersList`|List of fully connected layers in the network| `--layerList 784 120 40`|
|`expand_output`|How much we duplicate the output neurons (required for nudging a binary system)| `--expand_output 4`|

+ EqProp settings:

|Arguments|Description|Example|
|-------|------|------|
|`activationFun`|Activation function used in the network| `--activationFun heaviside`|
|`T`|Number of time steps for the free phase| `--T 20`|
|`Kmax`|Number of time steps for the nudge phase| `--Kmax 10`|
|`beta`|Nudging parameter|`--beta 2`|
|`gamma_neur`|Time step|`--gamma_neur 0.5`|
|`clamped`|Clamp the neurons states (1) or not (0)|`--clamped 1`|
|`rho_threshold`|Offset for the jump of the heaviside function|`--rho_threshold 0.5`|

+ Training settings:

|Arguments|Description|Example|
|-------|------|------|
|`dataset`|Selects which dataset to select to train the network on (mnist or digits)|`--dataset mnist`|
|`N_data`|Number of training images (for MNIST dataset)|`--N_data mnist`|
|`N_data_test`|Number of testing images (for MNIST dataset)|`--N_data_test mnist`|
|`lrW`|List of the learning for the biases|`--lrBias 1e-2 1e-2`|
|`weightClip`|Clamp the weights to an absolute given value|`--weightClip 1`|
|`lrB`|List of the learning for the biases|`--lrBias 1e-2 1e-2`|
|`biasClip`|Clamp the biases to an absolute given value|`--biasClip 1`|
|`batchSize`|Size of training mini-batches|`--batchSize 1`|
|`test_batchSize`|Size of testing mini-batches|`--test_batchSize 128`|
|`epochs`|Number of epochs to train the network|`--epochs 50`|

## Details about `Network.py`:
Each file contains a class of a network. Each class inherits from a nn.Module.

Each network has the following built-in functions:

+ `init`: specifies the parameter and the architecture of the network.

+ `get_bin_state`: return the binary state of a layer given the internal state at each time step

+ `stepper`: compute the update for the neurons state at a specific time step (Euler scheme)

+ `forward`: solve the dynamics of the system for a specific number of time steps (free or nudge phase)

+ `computeGrads`: compute the gradient of the parameters given the two equilibrium states

+ `updateWeight`: update all parameters in the network according to the gradient computed in computeGradient - vanilla SGD

+ `initHidden`: initialize the state of network before each free phase.
    
    
## Details about `Tools.py`:
Each file contains the same functions but adapted for each architecture.

+ `train_bin`: trains the network over the whole training dataset once. Run the free and nudge phase and update the weights. Track the number of changes in the weights.

+ `test_bin`: tests the network over the whole testing dataset once.

+ `initDataframe`: inits the dataframe where we store the training data.

+ `updateDataframe`: updates the same dataframe after each epoch: update train error, test error, number of changes of weights for each layer.

+ `createPath`: creates the path and the folder where the training data will be stored. Copy the plotFunction.py file in the same folder. Create a folder specific to the training 'S-X' in this general folder.

+ `saveHyperparameters`: creates a .txt file with all hyperparameters in the parser in the folder 'S-X'.

+ `generate_digits`: generate the digits dataset from sklearn.

+ `generate_mnist`: generate the MNIST/X dataset given the number of training and testing images specified in the parser.


## Commands to be run in the terminal to reproduce the results of the paper:
    ```
    python MLP/Determnistic/main.py --dataset mnist --N_data 1000 --N_data_test 100 --layersList 784 120 40 --expand_output 4 --lrW 0.1 0.1 --weightClip 1 -- --lrB 0.1 0.1 --biasClip 1 --batchSize 1 --test_batchSize 128 --epochs 50 --T 20 --Kmax 10 --beta 2  --activationFun heaviside --gamma_neur 0.5 --clamped 1 --rho_threshold 0.5 --device -1 --
	```

# Training the convolutional architecture

The code for training the convolutional architecture can be found in the folder ./Conv/QA-SA.

## Details about main.py

The parser accepts the following arguments:

+ Hardware settings:

|Arguments|Description|Example|
|-------|------|------|
|`simulated`| 0 if the D-Wave Ising machine is used - 1 if the Simulated annealing sampler is used| `--simulated 0`|

+ Architecture settings:

|Arguments|Description|Example|
|-------|------|------|
|`layersList`|Number of neurons in the output layer| `--layerList 4`|
|`expand_output`|How much we duplicate the output neurons (required for nudging a binary system)| `--expand_output 2`|
|`convList`|List of the conv layers with the numbers of channels specified - the number of channels of the input (1 for the patterns) has to be specified| `--convList 4 1`|
|`padding`|Padding applied for all the convolutions|`--padding 0`|
|`kernelSize`|Size of the kernel used for all the convolutions| `--kernelSize 2`|
|`Fpool`|Size of the pooling kernel| `--Fpool 2`|
|`pool_coef`|Coefficient used for the averaged pooling operation| `--pool_coef 0.25`|

+ EqProp settings:

|Arguments|Description|Example|
|-------|------|------|
|`mode`|Type of Ising problem to be submitted to the Ising machine - ising= {+/- 1 spins}, qubo: {0/1 spins}| `--mode ising`|
|`beta`|Nudging parameter| `-beta 5`|
|`n_iter_free`|Number of iterations for the free phase on a single data point| `--n_iter_free 10`|
|`n_iter_nudge`|Number of iterations for the nudge phase on a single data point| `--n_iter_nudge 10`|
|`frac_anneal_nudge`|Fraction of system non-annealed during the second phase| `--frac_anneal_nudge 0.25`|

+ Training settings:

|Arguments|Description|Example|
|-------|------|------|
|`dataset`|Dataset used to train the network| `--dataset patterns`|
|`lrWeightsFC`|Learning rate for the weights of the fully-connected classifier| `-lrWeightsFC 0.1`|
|`lrWeightsCONV`|Learning rate for the weights of the convolutional layer| `--lrWeightsCONV 0.1`|
|`lrBiasFC`|Learning rate for the biases of the fully-connected classifier| `-lrBiasFC 0.1`|
|`lrBiasCONV`|Learning rate for the biases of the convolutional layer| `--lrBiasCONV 0.1`|
|`batchSize`|Batch size| `--batchSize 1`|
|`epochs`|Number of epochs| `--epochs 20`|
|`chain_strength`|Value of the coupling in the chain of identical spins| `--chain_strength 2`|
|`auto_scale`|Enable (1) or disable (0) the auto-scale feature| `--auto_scale 0`|



## Details about `Network.py`:
Each file contains a class of a network. Each class inherits from a nn.Module.

Each network has the following built-in functions:

+ `init`: specifies the parameter and the architecture of the network.

+ `sample_to_s`: convert the sample obtained with the D-Wave Ising machine or the Simulated annealing sampler to tensors that match the dimension of each layer

+ `computeLossAcc`: compute the loss and if the network predict the good class for the input (accuracy) 

+ `computeGradients`: compute the instantaneous gradient for each parameter

+ `updateParams`: update all parameters in the network according to the gradient computed in computeGradient - vanilla SGD


## Details about `Tools.py`:
Each file contains the same functions but adapted for each architecture.

+ `train`: trains the network over the whole training dataset once. Run the free and nudge phase and update the weights. Track the number of changes in the weights.

+ `test`: tests the network over the whole testing dataset once.

+ `initDataframe`: inits the dataframe where we store the training data.

+ `updateDataframe`: updates the same dataframe after each epoch: update train error, test error, number of changes of weights for each layer.

+ `createPath`: creates the path and the folder where the training data will be stored. Copy the plotFunction.py file in the same folder. Create a folder specific to the training 'S-X' in this general folder.

+ `saveHyperparameters`: creates a .txt file with all hyperparameters in the parser in the folder 'S-X'.


## Commands to be run in the terminal to reproduce the results of the paper:
    ```
    python Conv/QA-SA/main.py --simulated 1 --dataset pattern --layersList 4 --expand_output 2 --convList 4 1 --padding 0 --kernelSize 2 --stride 1 --Fpool 2 --pool_coef 0.25 --lrWeightsFC 0.1 --lrWeightsCONV 0.1 --lrBiasFC 0.1 --lrBiasCONV 0.1 --batchSize 1 --epochs 20 --chain_strenght 2 --auto_scale 0
	```
