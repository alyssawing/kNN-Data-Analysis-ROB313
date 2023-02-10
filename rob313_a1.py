import numpy as np
import time
from data_utils import load_dataset
import matplotlib.pyplot as plt

###############################################################################
#############################   QUESTION 1  ###################################
###############################################################################

# k-NN algorithm for regression using 1 of 2 distance metrics
def knn_regression(x_train, y_train, x_test, k, distance_metric):
    '''kNN algorithm for regression with 2 distance metrics l1 and l2.
    k is estimated by 5-fold cross-validation. The distance metric is using 
    RMSE loss.'''

    if distance_metric == 'l1':
        # finding the Euclidian distance between test point and each training point:
        dist = np.sqrt(np.sum(np.square(x_train - x_test), axis=1)) # axis=1 means summing over rows
    elif distance_metric == 'l2':
        # Use Manhattan distance metric between test point and each training point:
        dist = np.sum(np.abs(x_train - x_test), axis=1) # axis=1 means summing over rows

    i_nn = np.argpartition(dist, kth=k)[:k] # indices (in training set) of the k nearest neighbours
    y_nn = y_train[i_nn] # target values of the nearest neighbours

    # prediction for the query point is the average of the kNN targets  (y_nn):
    y_hat = np.mean(y_nn)

    return [y_hat]    # return y-prediction for the x_test point

def rmse(y, y_hat):   #TODO: check what parameters are needed
    '''Find the root mean squared error (the mean of the errors of each point).'''
    return np.sqrt(np.mean(np.square(-y + y_hat)))

def cross_validation(x, y, model=knn_regression, v=5):
    '''Estimate the optimal k for kNN regression using v-fold cross-validation.
    The model is the estimator function to evaluate. The scoring will be the
    RMSE loss.'''

    n = len(x)  # number of data points
    partition_size = n // v  # number of data points in each partition
    index = np.arange(n)  # indices of each data point
    np.random.shuffle(index)  # shuffle the indices so partitions are randomly distributed
    folds = np.array_split(index, v)  # split the indices into v partitions

    rmses = [] # list of RMSE losses for each fold (to average over at the end)
    best_k = None # initialize optimal k value
    best_rmse = None # initialize optimal RMSE value (the lowest mean out of the 5)

    # For each fold, use the remaining v-1 folds to train the model and evaluate

    possible_k_indices = np.arange(1,10,1) # indices of the test k's
    for k in possible_k_indices:  # try k from 1 to 50 in steps of 5
        rmse_for_one_k = [] # list of RMSE losses for each fold (to average over at the end)
        for i in range(v):  # for each fold
            # Create the training and validation sets:
            validate_points =  folds[i] # indices of the data for validation
            x_val = x[validate_points] # validation data
            y_val = y[validate_points] # validation targets
            # train_points = np.concatenate((folds[:i] + folds[i+1:])) # indices of the training points TODO
            train_points = np.concatenate([folds[j] for j in range(v) if j != i]) # indices of the training points TODO
            x_train = x[train_points] # training data
            y_train = y[train_points] # training targets
            
            for distance_metric in ['l2']: # add in 'l2' later
                y_hats = []
                # Run kNN on each validation point and store the RMSE loss:
                for x_test_pt in x_val:
                    y_hats.append(model(x_train, y_train, x_test_pt, k, distance_metric))
                y_hats = np.array(y_hats)
                rmse_for_one_k.append(rmse(y_val, y_hats))

        rmse_for_one_k = np.array(rmse_for_one_k)
                # print(rmse_for_one_k, k)
        # Average the RMSE losses over each folds for one k:
        rmses.append(np.mean(rmse_for_one_k))

    rmses = np.array(rmses)
    print("RMSE loss averages for each k for l2: ", rmses)
    # Keep the minimum RMSE loss and the k that corresponds to it:
    min_rmse_index = np.argmin(rmses) # index of the minimum RMSE loss
    print("Lowest rmse error for l2: ", rmses[min_rmse_index])
    best_k = possible_k_indices[min_rmse_index] # k value that corresponds to the minimum RMSE loss

    return best_k

# def find_optimal_val(x_train, y_train, x_val, y_val, k_range, distance_metric): #TODO: check parameters??
#     '''Find the optimal k for kNN regression using validation data.
#     The scoring will be based on the RMSE loss.'''
#     # Use cross_validation on different values of k, store them and pick the 
#     # k value with the lowest validation error (lower returned mean error value)
#     pass

# Use validation error to pick k
# running multiple k/distance metric combos and picking the lowest validation error


if __name__ == '__main__':
    # Load the data:
    x_train, x_val, x_test, y_train, y_val, y_test = load_dataset('mauna_loa')
    # x_train, x_val, x_test, y_train, y_val, y_test = load_dataset('rosenbrock', n_train=5000, d=2)

    # Estimate the optimal k for kNN regression using 5-fold cross-validation:
    # k = cross_validation(x_train, y_train)

    # Fit the kNN model with the optimal k on the training data:
    # y_hats = []
    # for x_test_pt in x_val:
    #     y_hats.append(knn_regression(x_train, y_train, x_test_pt, 50, 'l1'))
    # y_hats = np.array(y_hats)

    # Evaluate the model on the test data:
    # print('The RMSE loss on the test data is: ', rmse(y_test, y_hats))

    # # Plot the predicted vs actual values:
    # #plt.scatter(y_test, y_hats)
    # plt.plot(x_train, y_train, 'o', label='Training Data')
    # plt.plot(x_train, y_hats, label='Test Data')
    # plt.xlabel('Actual')
    # plt.ylabel('Predicted')
    # plt.title('Predicted vs Actual Values')
    # plt.show()
    x = np.vstack([x_train, x_val])
    y = np.vstack([y_train, y_val])
    print("\nBest k found from cross validation: ", cross_validation(x, y))