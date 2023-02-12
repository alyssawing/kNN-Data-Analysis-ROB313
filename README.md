The `data_utils.py` module contains functions for loading, and viewing data.
Descriptions of the datasets can be found in the `PythonSetup.pdf` on Quercus.

The following code demonstrates how to load each of the datasets (note that `rosenbrock` is loaded differently since the number of training points and dimensionality must be specified manually).
```
from data_utils import load_dataset
x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mauna_loa')
x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('rosenbrock', n_train=5000, d=2)
x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('pumadyn32nm')
x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('iris')
x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mnist_small')
```
Each dataset has designated training, validation and testing splits (of course, the testing splits should never be viewed during model training or selection of hyperparameters, it is only for reporting your final generalization performance).
See the `data_utils.load_dataset` docstring for further details, if required.

The rob313_a1.py file can run as follows. In general, there are detailed comments
shown on each line for clarification as well. The question numbers are also clearly
emphasized in comment blocks, as well as the main function, where the different
datasets can be loaded.

For Question 1:

1. The code to uncomment can be found in the main function. If the plot is also
    desired, depending on which plot is wanted (Q1P1, Q1P2, or Q1P3), different
    code must be commented out. The code for plotting is clearly labelled - for 
    Q1P1, the code can be found within the cross-validation functionitself. For 
    the remaining plots as required (the orders also correspond to the order of 
    provided plots in the report), they can be found in the main function.
2. In the main function, different lines can be uncommented depending on which
    dataset is to be loaded. These are labelled by their dataset names.
3. To obtain the optimal k value and distance metric, you must manually change 
    the range of k values to be tested in the cross-validation function. Then,
    uncomment the lines where cross-validation is tested and run it on the 
    correct dataset. The terminal will print the minimum RMSE error for each 
    distance metric and provide which k corresponds to that. 
4. To obtain the actual test data, take this optimal k and distance metric
    and run the knn code in the for loop (to obtain each set of points for
    y_hats). The rmse function can be run as well to obtain the testing loss.

For Question 2: 

1. Uncomment all of the code in the question 2 block in the main function.
2. You can adjust the different dimension (d) values to test the data over,
    and the plt.plot block of code will plot the differences in elapsed time
    to compare the k-NN with either brute force or k-d tree data structure
    method of computing the nearest neighbours. (This will return the same
    plot that is provided in the report).

For Question 3:

1. Uncomment all of the code in the main function for Question 3. However, only
    import one of the two classification datasets.
2. The q3_estimator function runs to estimate, using accuracy, the best k and
    distance metric. k-NN classification is then run  on those optimal values
    and test data to return the accuracy of the model on the test data.

For Question 4:

1. In the first block of code in the main function for Question 4, uncomment the
    correct dataset that is desired to run. 
2. If this dataset is a classification dataset, uncomment the code below the 
    comment that says "FOR CLASSIFICATION," which turns the one-hot encoding
    into a numbered list.
3. If the dataset is mauna_loa, uncomment the bottom chunk of code ("For mauna_loa
    heading) to obtain the plot of the test predictions for linear regression.
    This is the same graph that is provided in the report, and it can be 
    useful to compare it to the plot of mauna_loa with the k-NN model.


