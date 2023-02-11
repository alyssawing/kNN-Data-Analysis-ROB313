import numpy as np
import time
from data_utils import load_dataset
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

###############################################################################
#############################   QUESTION 1  ###################################
###############################################################################

# k-NN algorithm for regression using 1 of 2 distance metrics
def knn_regression(x_train, y_train, x_test, k, distance_metric):
    '''kNN algorithm for regression with 2 distance metrics l1 and l2.
    k is estimated by 5-fold cross-validation. The distance metric is using 
    RMSE loss.'''

    if distance_metric == 'l2':
        # finding the Euclidian distance between test point and each training point:
        dist = np.sqrt(np.sum(np.square(x_train - x_test), axis=1)) # axis=1 means summing over rows
    elif distance_metric == 'l1':
        # Use Manhattan distance metric between test point and each training point:
        dist = np.sum(np.abs(x_train - x_test), axis=1) # axis=1 means summing over rows

    i_nn = np.argpartition(dist, kth=k)[:k] # indices (in training set) of the k nearest neighbours
    y_nn = y_train[i_nn] # target values of the nearest neighbours
    y_hat = np.mean(y_nn) # prediction for the query point is the average of the kNN targets  (y_nn)

    return [y_hat]    # return y-prediction for the x_test point

def rmse(y, y_hat):   
    '''Find the root mean squared error (the mean of the errors of each point).'''
    return np.sqrt(np.mean(np.square(y - y_hat)))

def cross_validation(x, y, distance_metric, model=knn_regression, v=5):
    '''Estimate the optimal k for kNN regression using v-fold cross-validation.
    The model is the estimator function to evaluate. The scoring will be the
    RMSE loss, and nearest neighbours are brute force'''

    n = len(x)  # number of data points
    partition_size = n // v  # number of data points in each partition
    index = np.arange(n)  # indices of each data point
    np.random.shuffle(index)  # shuffle the indices so partitions are randomly distributed
    folds = np.array_split(index, v)  # split the indices into v partitions

    rmses = [] # list of RMSE losses for each fold (to average over at the end)
    best_k = None # initialize optimal k value
    y_all_preds = np.array([]) # initialize array of all y values for plotting cross-validation prediction curves 

    # For each fold, use the remaining v-1 folds to train the model and evaluate
    possible_k_indices = np.arange(1,10,1) # indices of the test k's
    for k in possible_k_indices:  # try k from 1 to 50 in steps of 5
        print("testing k = ", k)
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
            y_hats = []
            # Run kNN on each validation point and store the RMSE loss:
            for x_test_pt in x_val:
                y_hats.append(model(x_train, y_train, x_test_pt, k, distance_metric))
            y_hats = np.array(y_hats)
            # print("Shape of y_hats: ", y_hats.shape)
            y_all_preds = np.concatenate(y_all_preds + y_hats) # for plotting cross-validation prediction curves
            rmse_for_one_k.append(rmse(y_val, y_hats))

        # plot the prediction curves for each value of k:
        plt.plot(x_val, y_hats, 'o', label='k = ' + str(k))
        plt.xlabel('x values')
        plt.ylabel('Predicted y values')
        plt.title('Cross-validation prediction curves: moana_loa dataset')
        plt.legend()


        rmse_for_one_k = np.array(rmse_for_one_k)
                # print(rmse_for_one_k, k)
        rmses.append(np.mean(rmse_for_one_k))   # Average the RMSE losses over each folds for one k
    plt.show()
    rmses = np.array(rmses)
    # print("RMSE loss averages for each k: ", rmses)

    # Keep the minimum RMSE loss and the k that corresponds to it:
    min_rmse_index = np.argmin(rmses) # index of the minimum RMSE loss
    print("Lowest rmse error: ", rmses[min_rmse_index])
    best_k = possible_k_indices[min_rmse_index] # k value that corresponds to the minimum RMSE loss

    # # For plotting RMSE loss vs. k value in mauna_loa:
    # plt.plot(possible_k_indices, rmses, 'o-')    
    # plt.xlabel('k values')
    # plt.ylabel('RMSE loss')
    # plt.title('RMSE loss for each k value: moana_loa dataset')
    # plt.show()

    # # For plotting cross-validation prediction curves in mauna_loa:
    # plt.plot(x, y_all_preds, 'o', label='true values')
    # plt.xlabel('x values')
    # plt.ylabel('Predicted y values')
    # plt.title('Cross-validation prediction curves: moana_loa dataset')
    # plt.show()

    # For plotting the prediction on the test set in mauna_loa:


    return best_k

###############################################################################
#############################   QUESTION 2  ###################################
###############################################################################

def euclidean_distance(x_train, x_test):
    '''Find the Euclidian distance between the test point and each training point.'''
    return np.sqrt(np.sum(np.square(x_train - x_test), axis=1)) # axis=1 means summing over rows

def knn_regression_kd(x_train, y_train, x_test, k, distance_metric=euclidean_distance):
    '''kNN algorithm for regression with 2 distance metrics l1 and l2.
    k is estimated by 5-fold cross-validation. The distance metric is using 
    RMSE loss, and nearest neighbours are found using a kd tree.'''

    # Use a kd tree to find nearest neighbours:
    tree = KDTree(x_train, metric='euclidean')
    dist, i_nn = tree.query(x_test, k=k) # indices (in training set) of the k nearest neighbours
    y_nn = y_train[i_nn] # target values of the nearest neighbours for the query point
    y_hat = np.mean(y_nn, axis=1) # prediction for the query point is the average of the kNN targets (y_nn)
    
    return [y_hat]    # return y-prediction for the x_test point

###############################################################################
#############################   QUESTION 3  ###################################
###############################################################################

def knn_classification(x_train, y_train, x_test, k, distance_metric):
    '''kNN algorithm for classification with 2 distance metrics l1 and l2.
    k is estimated by maximizing accuracy on the validation split. The distance 
    metric is using RMSE loss, and nearest neighbours are found using a kd tree.'''
    if distance_metric == 'l2': # Euclidian distance
        dist = np.sqrt(np.sum(np.square(x_train - x_test), axis=1)) # axis=1 means summing over rows TODO problem here!!
    elif distance_metric == 'l1':   # Manhattan distance
        dist = np.sum(np.abs(x_train - x_test), axis=1) # axis=1 means summing over rows

    # Use a kd tree to find nearest neighbours:
    tree = KDTree(x_train, metric='euclidean')
    dist, i_nn = tree.query(x_test, k=k) # indices (in training set) of the k nearest neighbours
    y_nn = y_train[i_nn] # target values of the nearest neighbours for the query point
    y_hat = np.mean(y_nn, axis=1) # prediction for the query point is the average of the kNN targets (y_nn)
    
    return [y_hat]    # return y-prediction for the x_test point

def q3_estimator(x_train, y_train, x_val, y_val, distance_metric):
    '''Estimate the best k for kNN classification and the best distance metric 
    by maximizing the accuracy (fraction of correct predictions) on the 
    validation split. Return the best k and the best distance metric.'''
    best_k, best_metric = None, None # initialize
    best_accuracy = 0 # initialize
    pass # TODO


if __name__ == '__main__':

    #===========================    Q1   ===================================#

    # # Load the data:
    x_train, x_val, x_test, y_train, y_val, y_test = load_dataset('mauna_loa')
    print("Dataset being tested: mauna_loa\n\n")
    # x_train, x_val, x_test, y_train, y_val, y_test = load_dataset('rosenbrock', n_train=5000, d=2)
    # print("Dataset being tested: rosenbrock\n\n")
    # x_train, x_val, x_test, y_train, y_val, y_test = load_dataset('pumadyn32nm')
    # print("Dataset being tested: pumadyn32nm\n\n")
    # x_train, x_val, x_test, y_train, y_val, y_test = load_dataset('iris')
    # print("Dataset being tested: iris\n\n")
    # x_train, x_val, x_test, y_train, y_val, y_test = load_dataset('mnist_small')
    # print("Dataset being tested: mnist_small\n\n")

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
    # print("\nBest k found from cross validation (l1): ", cross_validation(x, y, 'l1'))
    print("\nBest k found from cross validation (l2): ", cross_validation(x, y, 'l2'))


    #===========================    Q2   ===================================#

    # dimensions = []
    # times = [] # time taken to run knn_regression_kd for each dimension

    # for d in range(2, 101, 2): # test for dimensions 2, 4, ..., 100
    #     print("Testing dimension: ", d)
    #     x_train, x_val, x_test, y_train, y_val, y_test = load_dataset('rosenbrock', n_train=5000, d=d)
    #     time1 = time.time()
    #     knn_regression_kd(x_train, y_train, x_test, 5, euclidean_distance)
    #     time2 = time.time()
    #     dimensions.append(d)
    #     times.append(time2 - time1)
    
    # plt.plot(dimensions, times, 'o-')
    # plt.xlabel('Dimensions')
    # plt.ylabel('Time taken to run k-NN')
    # plt.title('Time taken to run k-NN for each dimension')
    # plt.show()
