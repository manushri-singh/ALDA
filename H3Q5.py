from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import pylab

import os

# (1) Construct neural networks using the given training dataset (X train, Y train) using different number of hidden
#  neurons. Set the parameters as follows: activation function for hidden layer='relu', activation for output layer
# =sigmoid, loss function =mse, metrics= 'accuracy', epochs=5, batch size=1. For each model, change the number
# of hidden neurons in the order of 2, 4, 6, 8, 10.
# (2) Validate each neural network using the given validation dataset (X val, Y val). The validation accuracy is used
# to determine how many number of hidden neurons are optimal for this problem. Provide the core code for "neural
# network learning" with comments in your report. (Please apply a fixed random seed 7 in order to generate a
# same result every time.)

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# define base model
def baseline_model(num):
    # create model
    np.random.seed(7)
    model = Sequential()
    ##Input Layer
    # model.add(Dense(1, input_dim=64, init='normal'))
    ##Hidden Layer
    model.add(Dense(num, input_dim=64, activation='relu'))
    ## Output Layer
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

# load datasets
Xtrain = np.genfromtxt('data/hw3q5/X_train.csv', delimiter=',', dtype=None)
Ytrain = np.genfromtxt('data/hw3q5/Y_train.csv', delimiter=',', dtype=None)
Xval = np.genfromtxt('data/hw3q5/X_val.csv', delimiter=',', dtype=None)
Yval = np.genfromtxt('data/hw3q5/Y_val.csv', delimiter=',', dtype=None)
Xtest = np.genfromtxt('data/hw3q5/X_test.csv', delimiter=',', dtype=None)
Ytest = np.genfromtxt('data/hw3q5/Y_test.csv', delimiter=',', dtype=None)


trainaccarr=[]
valaccarr=[]
hidden=[2, 4, 6, 8, 10]

for i in hidden:
    print('\n')
    model=baseline_model(i)
    model_info = model.fit(Xtrain, Ytrain, batch_size=1, epochs=5, verbose=1,validation_data=(Xval, Yval))
    (trainloss, trainacc) = model.evaluate(Xtrain, Ytrain, verbose=1)
    (valloss,valacc) = model.evaluate(Xval, Yval, verbose=1)
    trainaccarr.append(trainacc)
    valaccarr.append(valacc)
    print('\n')

# (c) (3 points) Plot a figure, where the horizontal x-axis is the number of hidden neurons, and the vertical y-axis
# is the accuracy. Please plot both training and validation accuracy in your figure. (Note that the exact accuracy
# could be slightly different according to your working environments, however you can analyze the trend.)

Xaxis=hidden
pylab.plot(Xaxis, trainaccarr, '-xr', label='Training Accuracy')
pylab.plot(Xaxis, valaccarr, '-xb', label='Validation Accuracy')
pylab.legend(loc='upper left')
pylab.xlabel('Number of Hidden Neurons')
pylab.ylabel('Accuracy')
pylab.title('Training_vs_Validation_Accuracy')
pylab.savefig('Training_vs_Validation_Accuracy.png')


optimal=valaccarr.index(max(valaccarr))
testmodel = baseline_model(hidden[optimal])
testmodel_info = testmodel.fit(Xtrain, Ytrain, batch_size=1, epochs=5, verbose=1, validation_data=(Xval, Yval))
(finaltestloss, finaltestacc) = testmodel.evaluate(Xtest, Ytest, verbose=1)
print('\n\n\n')
print('\n Testing Accuracy is: ')
print(finaltestacc)
