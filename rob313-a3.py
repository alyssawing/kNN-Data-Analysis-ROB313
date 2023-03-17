import numpy as np
import time
from data_utils import load_dataset
import matplotlib.pyplot as plt
import math

##############################################################################
#######################      QUESTION 1      #################################
##############################################################################

# full-batch gradient descent
def full_batch_gd(learningrates, epochs=25, validation=True, dataset='puma'):
    '''Conduct full-batch gradient descent on the pumadyn32nm dataset, using 
    the first 1000 points in the training set to predict on the test set. 
    Calculate the final test RMSE. The parameters are:
    - learningrates: a list of the learning rates to test and plot
    - epochs: the number of epochs to run (default is 25)
    - dataset: the dataset to use (default is pumadyn32nm)
    - validation: whether to use validation (default is True) or the test set

    The weights are initialized to 0, and plot thhe exact (full-batch) loss vs. 
    iteration number using a range of learning rates. Indicate the exact value of
    the optimum on all plots. Select the best learning rate for each method (and
    beta for SGD+momentum).)
    '''
    # load the dataset
    x_train, x_val, x_test, y_train, y_val, y_test = load_dataset('pumadyn32nm')
    print("Dataset being tested: pumadyn32nm\n\n")

    # only use the first 1000 points of the training set (for time's sake):
    x_train = x_train[:1000]
    y_train = y_train[:1000]

    if validation==True: # to find the validation RMSEs
        print("Using validation set to find best model\n\n")
        x = x_val
        y = y_val
        filler = "validation" # for plotting purposes
    else:   # use x_test, y_test instead of x_val, y_val to find test RMSEs
        print("Using test set to find best model\n\n")
        x = x_test
        y = y_test
        filler = "test" # for plotting purposes

    # initialize weights, a column vector the same size as the number of features + 1 for bias (adds a row):
    weights = np.zeros((x_train.shape[1]+1, 1))
    N = len(x_train) # N is the number of training points

    # increase dimension of x by 1 to account for bias:
    x_train = np.hstack((x_train, np.ones((x_train.shape[0], 1))))
    x = np.hstack((x, np.ones((x.shape[0], 1))))
    rmse_vt = [] # list to store rmse from validation or test set (depending on input to function)

    # apply gradient descent over a ranage of learning rates:
    for lr in learningrates:
        rmse_train = []

        for e in range(epochs):
            # print("shape of x_train: ", x_train.shape)
            # print("shape of weights: ", weights.shape)

            # calculate the gradient of the least squared loss (objective function) over the dataset:
            y_hat = np.dot(x_train, weights)
            gradL = -2/N * np.dot(x_train.T, (y_train - y_hat)) #TODO double check bias 

            # update the weights and bias:
            weights = weights - lr * gradL

            # calculate the RMSE on the validation set:
            rmse_train.append(get_rmse(x_train, y_train, weights))

        # plot the RMSE vs. epoch number:
        plt.plot(range(epochs), rmse_train, label='lr = {}'.format(lr))

        # indicate the value of the exact optimum point on the plot:
        plt.plot(np.argmin(rmse_train), np.min(rmse_train), 'ro')

        # find the rmse on the validation or test set (depending on input to function):
        y_hat_vt = np.dot(x, weights)
        rmse_vt.append(np.sqrt(np.mean((y - y_hat_vt)**2)))
    
    # show the plot of epoch vs. rmse on training set
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('RMSE vs. Epoch for Full-Batch Gradient Descent: Training set')
    plt.legend()
    plt.show()

    # plot the rmse on validation/test set vs. learning rate:
    plt.plot(learningrates, rmse_vt, 'mo-')
    # plot the minimum rmse on validation/test set vs. learning rate:
    plt.plot(learningrates[np.argmin(rmse_vt)], np.min(rmse_vt), 'ro')
    plt.xlabel('Learning Rate')
    plt.ylabel('RMSE')
    plt.title('RMSE vs. Learning Rate for Full-Batch Gradient Descent: ' + filler + ' set')
    plt.show()

def get_rmse(x, y, weights):
    '''Given the weights model, use x to predict what y should be (y_hat). Then
    return the RMSE of the model's predictions given the true y
    '''
    # calculate the RMSE (y_hat = x * weights)):
    rmse = np.sqrt(np.mean((y - np.dot(x, weights))**2))
    return rmse

# stochastic gradient descent
def sgd(weights, bias, mbs, validation=True, momentum=False, dataset='puma'):
    '''Conduct stochastic gradient descent on the pumadyn32nm dataset, using 
    the first 1000 points in the training set to predict on the test set. 
    Calculate the final test RMSE. The parameters are:
    - weights: the initial weights (initialized to 0)
    - bias: the initial bias
    - mbs: the mini-batch size
    - momentum: whether to use momentum (default is False)
    - dataset: the dataset to use (default is pumadyn32nm)
    - validation: whether to use validation (default is True) or the test set

    The weights are initialized to 0, and plot thhe exact (full-batch) loss vs. 
    iteration number using a range of learning rates. Indicate the exact value of
    the optimum on all plots. Select the best learning rate for each method (and
    beta for SGD+momentum).)
    '''
    # load the dataset
    x_train, x_val, x_test, y_train, y_val, y_test = load_dataset('pumadyn32nm')
    print("Dataset being tested: pumadyn32nm\n\n")

    if validation==True: # to find the validation RMSEs
        print("Using validation set to find best model\n\n")
        x = x_val
        y = y_val
    else:   # use x_test, y_test instead of x_val, y_val to find test RMSEs
        print("Using test set to find best model\n\n")
        x = x_test
        y = y_test

##############################################################################
#######################         MAIN        ##################################
##############################################################################

if __name__ == "__main__":

    #########################       Q1      ###################################

    print("******************* Q1 *******************")

    # full-batch gradient descent
    print("******************* full batch *******************")
    learningrates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
    full_batch_gd(learningrates, epochs=25, validation=True, dataset='puma')