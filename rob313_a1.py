import numpy as np
import time
from data_utils import load_dataset
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

###############################################################################
#############################   QUESTION 1  ###################################
###############################################################################

# k-NN algorithm for regression using 1 of 2 distance metrics
def knn_regression(x_train, y_train, x_test, k, distance_metric='l2'):
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
    possible_k_indices = np.arange(1,50,2) # indices of the test k's
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

        # PLOT THE PREDICTION CURVES FOR EACH VALUE OF K (Q1 PART 1):
        # plt.plot(x_val, y_hats, 'o', markersize=3, label='k = ' + str(k))
        # plt.xlabel('x values')
        # plt.ylabel('Predicted y values')
        # plt.title('Cross-validation prediction curves: moana_loa dataset')
        # plt.legend()


        rmse_for_one_k = np.array(rmse_for_one_k)
                # print(rmse_for_one_k, k)
        rmses.append(np.mean(rmse_for_one_k))   # Average the RMSE losses over each folds for one k
    # plt.show()
    rmses = np.array(rmses)
    # print("RMSE loss averages for each k: ", rmses)

    # Keep the minimum RMSE loss and the k that corresponds to it:
    min_rmse_index = np.argmin(rmses) # index of the minimum RMSE loss
    print("Lowest rmse error: ", rmses[min_rmse_index])
    best_k = possible_k_indices[min_rmse_index] # k value that corresponds to the minimum RMSE loss

    # # PLOTTING RMSE LOSS VS. K VALUE IN MAUNA_LOA (Q1 PART 3): 
    # plt.plot(possible_k_indices, rmses, 'o-')    
    # plt.xlabel('k values')
    # plt.ylabel('RMSE loss')
    # plt.title('RMSE loss for each k value: mauna_loa dataset')
    # plt.show()

    return best_k

###############################################################################
#############################   QUESTION 2  ###################################
###############################################################################

# def euclidean_distance(x_train, x_test):
#     '''Find the Euclidian distance between the test point and each training point.'''
#     return np.sqrt(np.sum(np.square(x_train - x_test), axis=1)) # axis=1 means summing over rows

def knn_regression_kd(x_train, y_train, x_test, k):
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

# def manhattan_distance(x_train, x_test):
#     '''Find the Manhattan distance between the test point and each training point.'''
#     return np.sum(np.abs(x_train - x_test), axis=1) # axis=1 means summing over rows

def knn_classification(x_train, y_train, x_test, k, distance_metric='l2'):
    '''kNN algorithm for classification with 2 distance metrics l1 and l2.
    k is estimated by maximizing accuracy on the validation split. The distance 
    metric is using RMSE loss, and nearest neighbours are found using a kd tree.'''
    if distance_metric == 'l2': # Euclidian distance
        tree = KDTree(x_train, metric='euclidean')  # Use a kd tree to find nearest neighbours
    elif distance_metric == 'l1':   # Manhattan distance
        tree = KDTree(x_train, metric='manhattan')

    y_hats = []
    dist, i_nn = tree.query(x_test, k=k) # indices (in training set) of the k nearest neighbours
    y_nn = y_train[i_nn] # target values of the nearest neighbours for the query point

    for i in y_nn: # loop through all lists of neighbour points
        vote, count = np.unique(i, return_counts=True, axis=0)
        y_hats.append(vote[np.argmax(count)])
    
    y_hats = np.array(y_hats)

    return y_hats    # return y-prediction labels for x test point

def find_accuracy(y_val, y_hats):
    '''Find the accuracy (fraction of correct predictions) on the validation 
    split.'''
    correct = 0

    correct = np.sum(y_val == y_hats)    

    # print("y_val shape: ", y_val.shape)
    return correct / y_val.shape[0]

def q3_estimator(x_train, y_train, x_val, y_val):
    '''Estimate the best k for kNN classification and the best distance metric 
    by maximizing the accuracy (fraction of correct predictions) on the 
    validation split. Return the best k and the best distance metric.'''
    best_k, best_metric = None, None # initialize
    best_accuracy = 0 # initialize
    possible_k_indices = np.arange(1,6,1) # possible k values
    possible_metrics = ['l1', 'l2'] # possible distance metrics

    for k in possible_k_indices:
        print("Testing k: ", k, "\n")
        for metric in possible_metrics:
            y_hats = knn_classification(x_train, y_train, x_val, k, metric)
            accuracy = find_accuracy(y_val, y_hats)
            # print("accuracy: ", accuracy, ",   k: ", k, ",   metric: ", metric, "\n")
            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                # print("Updated accuracy: ", best_accuracy, ",   k: ", k, ",   metric: ", metric, "\n")
                best_k = k
                best_metric = metric

    return best_accuracy, best_k, best_metric

###############################################################################
#############################   QUESTION 4  ###################################
###############################################################################


###############################################################################
#############################   MAIN  #########################################
###############################################################################

if __name__ == '__main__':

    #===========================    Q1   ===================================#

    # # Load the data:
    # x_train, x_val, x_test, y_train, y_val, y_test = load_dataset('mauna_loa')
    # print("Dataset being tested: mauna_loa\n\n")
    # x_train, x_val, x_test, y_train, y_val, y_test = load_dataset('rosenbrock', n_train=5000, d=2)
    # print("Dataset being tested: rosenbrock\n\n")
    # x_train, x_val, x_test, y_train, y_val, y_test = load_dataset('pumadyn32nm')
    # print("Dataset being tested: pumadyn32nm\n\n")


    # x = np.vstack([x_train, x_val])
    # y = np.vstack([y_train, y_val])
    # # print("\nBest k found from cross validation (l1): ", cross_validation(x, y, 'l1'))
    # print("\nBest k found from cross validation (l2): ", cross_validation(x, y, 'l2'))

    # PLOTTING THE PREDICTION ON THE TEST SET IN MAUNA_LOA (Q1 PART 2): TODO
    # y_hats = []
    # for x_test_pt in x_test:
    #     y_hat = knn_regression(x_train, y_train, x_test_pt, 2,'l2')
    #     y_hats.append(y_hat)
    # # print("y_hats: ", y_hats)
    # plt.plot(x_test, y_test, 'o', markersize=3, label='test set')
    # plt.plot(x_test, y_hats, 'o', markersize=3, label='k-NN predictions')
    # plt.plot(x_train, y_train, 'o', markersize=2, label='training set')
    # plt.xlabel('x values')
    # plt.ylabel('Predicted y values')
    # plt.title('Cross-validation prediction curves: mauna_loa dataset')
    # plt.legend()
    # plt.show()


    #===========================    Q2   ===================================#

    # x_train, x_val, x_test, y_train, y_val, y_test = load_dataset('rosenbrock', n_train=5000, d=2)
    # print("Dataset being tested: rosenbrock\n\n")
    # x = np.vstack([x_train, x_val])
    # y = np.vstack([y_train, y_val])

    # dimensions = []
    # kd_times = [] # time taken to run knn_regression_kd with k-d trees for each dimension
    # bf_times = [] # time taken to run knn_regression with brute force for each dimension
    # bf_y_hats = [] # y_hats for brute force

    # for d in range(2, 101, 2): # test for dimensions 2, 4, ..., 100
    #     print("Testing dimension: ", d)
    #     x_train, x_val, x_test, y_train, y_val, y_test = load_dataset('rosenbrock', n_train=5000, d=d)
    #     time1 = time.time()
    #     knn_regression_kd(x_train, y_train, x_test, 5)
    #     time2 = time.time()

    #     time3 = time.time()
    #     for x_test_pt in x_test:
    #         y_hat = knn_regression(x_train, y_train, x_test_pt, 5, 'l2')
    #         bf_y_hats.append(y_hat)
    #     time4 = time.time()

    #     dimensions.append(d)
    #     kd_times.append(time2 - time1)
    #     bf_times.append(time4 - time3)
    
    # plt.plot(dimensions, kd_times, 'o-', label='k-d trees')
    # plt.plot(dimensions, bf_times, 'o-', label='brute force')
    # plt.xlabel('Dimensions')
    # plt.ylabel('Time taken to run k-NN')
    # plt.title('Time taken to run k-NN with k-d trees vs brute force for each dimension')
    # plt.legend()
    # plt.show()

    #===========================    Q3   ===================================#

    # x_train, x_val, x_test, y_train, y_val, y_test = load_dataset('iris')
    # print("Dataset being tested: iris\n\n")
    x_train, x_val, x_test, y_train, y_val, y_test = load_dataset('mnist_small')
    print("Dataset being tested: mnist_small\n\n")
    
    # Transforming 1-hot encoded into regular list
    # for example: [[0],[2]] means the first element is true, then the second element is true...
    y_train = np.argmax(y_train, axis=1)
    y_train = y_train.reshape(-1, 1)
    y_valid = np.argmax(y_val, axis=1)
    y_valid = y_valid.reshape(-1, 1)
    y_test = np.argmax(y_test, axis=1)
    y_test = y_test.reshape(-1, 1)

    best_accuracy, best_k, best_metric = q3_estimator(x_train, y_train, x_val, y_valid)
    print("Best accuracy found: ", best_accuracy)
    print("Best k found: ", best_k)
    print("Best metric found: ", best_metric)

    #===========================    Q4   ===================================#


