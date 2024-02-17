from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import configparser
from Logs import logging as logs

def trainHyperParameters(model, grid, train_x, train_y):
    grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(train_x, train_y)
    return grid_search.best_estimator_

# Run Random Forest Regressor
# Inputs:
    # train_x
    # train_y
    # test_x
    # test_y
def runRFR(train_x, train_y, test_x, test_y):
    rfr = RandomForestRegressor()
    param_grid = {
        'n_estimators': [100, 200, 400, 800, 1200],
        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'criterion': ['squared_error','absolute_error','friedman_mse','poisson'],
        'min_samples_split': [2, 4, 8, 12],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    best_rfr = trainHyperParameters(rfr, param_grid, train_x, train_y)
    best_rfr.fit(train_x, train_y)

    #importances = rfr.feature_importances_
    predictions = best_rfr.predict(test_x)
    directionScore, distanceScore = measurePerformance(predictions, test_y)

    return directionScore, distanceScore


def trainHyperParameters(model, grid, train_x, train_y):
    grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(train_x, train_y)
    return grid_search.best_estimator_

# Run Random Forest Regressor
# Inputs:
    # train_x
    # train_y
    # test_x
    # test_y
def runRFR(train_x, train_y, test_x, test_y):
    rfr = RandomForestRegressor()
    param_grid = {
        'n_estimators': [100, 200, 400, 800, 1200],
        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'criterion': ['squared_error','absolute_error','friedman_mse','poisson'],
        'min_samples_split': [2, 4, 8, 12],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    best_rfr = trainHyperParameters(rfr, param_grid, train_x, train_y)
    best_rfr.fit(train_x, train_y)

    #importances = rfr.feature_importances_
    predictions = best_rfr.predict(test_x)
    directionScore, distanceScore = measurePerformance(predictions, test_y)

    return directionScore, distanceScore


config = configparser.ConfigParser()
config.read('Data//config.ini')
# Run Decision Tree Training and Testing
# Inputs:
    # train_x
    # train_y
    # test_x
    # test_y
    # max_depth (maximum depth of tree)
    # max_features (number of features considered when looking for best split)
    # max_leaf_nodes (Maximum number of leaf nodes in the tree)
# Ouput:
    # Decision Tree Model, Training Accuracy, Testing Accuracy
def runDT(train_x, train_y, test_x, test_y, max_depth, max_features, max_leaf_nodes):
    if (config['MODELS']['DTC'] == False):
        return None, None, None
    dt = DecisionTreeClassifier(max_depth=max_depth, max_features=max_features, max_leaf_nodes=max_leaf_nodes, class_weight='balanced')
    #dt = DecisionTreeClassifier()   
    # Train Model
    print("training decision tree model...")
    dt.fit(train_x, train_y)
    print("done!")

    # Model Statistics
    print("getting statistics...\n")

    y_trainPred = dt.predict(train_x)
    trainStats = get_infield_statistics(train_y, y_trainPred)[0]
    train_accuracy = trainStats[0]
    train_averageError =  trainStats[1]

    y_pred = dt.predict(test_x)
    testStats = get_infield_statistics(test_y, y_pred)[0]
    test_accuracy = testStats[0]
    test_averageError = testStats[1]

    train_stat = [train_accuracy, train_averageError]
    test_stat = [test_accuracy, test_averageError]

    # CONFIG (add debug/print mode)
    logs.logModel("DecisionTree", dt, train_stat, test_stat, [train_x, train_y, test_x, test_y, y_trainPred, y_pred],
                   ["Max Tree Depth: ", max_depth, "Max Tree Features: ", max_features, "Max Leaf Nodes: ", max_leaf_nodes])
    
    logs.printModel("DecisionTree", dt, train_stat, test_stat, [train_x, train_y, test_x, test_y, y_trainPred, y_pred],
                   ["Max Tree Depth: ", max_depth, "Max Tree Features: ", max_features, "Max Leaf Nodes: ", max_leaf_nodes])
    
    print("done!")

    return dt, train_accuracy, test_accuracy


# Run Naive Bayes Training and Testing
# Inputs:
    # train_x
    # train_y
    # test_x
    # test_y
    # var_smoothing (ammount of smoothing in the model: 1e-7, 1e-8, 1e-9 [default], 1e-10, 1e-11)
# Ouput:
    # Naive Bayes Model, Training Accuracy, Testing Accuracy
def runNB(train_x, train_y, test_x, test_y, var_smoothing):
    if (config['MODELS']['NB'] == False):
        return None, None, None
    nb = GaussianNB(var_smoothing=var_smoothing) #class_weight='balanced'
    
    # Train Model
    print("training Naive Bayes model...")
    nb.fit(train_x, train_y)
    print("done!")

    # Model Statistics
    print("getting statistics...")

    y_trainPred = nb.predict(train_x)
    trainStats = get_infield_statistics(train_y, y_trainPred)
    train_accuracy = trainStats[0]
    train_averageError = trainStats[1]

    y_pred = nb.predict(test_x)
    testStats = get_infield_statistics(test_y, y_pred)
    test_accuracy = testStats[0]
    test_averageError = testStats[1]

    train_stat = [train_accuracy, train_averageError]
    test_stat = [test_accuracy, test_averageError]
    
    logs.logModel("NaiveBayes", nb, train_stat, test_stat, [train_x, train_y, test_x, test_y, y_trainPred, y_pred],
                   ["Var Smoothing: ", var_smoothing])
    
    logs.printModel("NaiveBayes", nb, train_stat, test_stat, [train_x, train_y, test_x, test_y, y_trainPred, y_pred],
                   ["Var Smoothing: ", var_smoothing])
    
    print("done!")

    return nb, train_accuracy, test_accuracy

# Run Logistic Regression Training and Testing
# Inputs:
    # train_x
    # train_y
    # test_x
    # test_y
    # lr: learning rate (0 to 1.0)
    # e: epochs (iterations)
# Ouput:
    # Logistic Regression Model, Training Accuracy, Testing Accuracy
def runLogReg(train_x, train_y, test_x, test_y, lr, e):
    if (config['MODELS']['LR'] == False):
        return None, None, None
    
    logreg = LogisticRegression(C=lr, max_iter=e, class_weight='balanced')

    print("training logistic regression model...")
    logreg.fit(train_x, train_y)
    print("done!")

    # Model Statistics
    print("getting statistics...")

    y_trainPred = logreg.predict(train_x)
    trainStats = get_infield_statistics(train_y, y_trainPred)
    train_accuracy = trainStats[0]
    train_averageError = trainStats[1]

    y_pred = logreg.predict(test_x)
    testStats = get_infield_statistics(test_y, y_pred)
    test_accuracy = testStats[0]
    test_averageError = testStats[1]

    train_stat = [train_accuracy, train_averageError]
    test_stat = [test_accuracy, test_averageError]
    
    print("logging statistics...")
    logs.logModel("LogisticRegression", logreg, train_stat, test_stat, [train_x, train_y, test_x, test_y, y_trainPred, y_pred],
                   ["Learning Rate: ", lr, "Epochs: ", e])
    
    logs.printModel("LogisticRegression", logreg, train_stat, test_stat, [train_x, train_y, test_x, test_y, y_trainPred, y_pred],
                   ["Learning Rate: ", lr, "Epochs: ", e])
    
    print("done!")

    return logreg, train_accuracy, test_accuracy

# Run Support Vector Machine Training and Testing
# Inputs:
    # train_x
    # train_y
    # test_x 
    # test_y
    # rC: regularization constant
    # kernel: Kernel type (can be 'linear', 'poly', 'rbf', 'sigmoid')
    # degree: Degree of the polynomial kernel function (ignored by all other kernels)
    # gamma:  Kernel coefficient for 'rbf', 'poly', and 'sigmoid' ('scale')
    # coef0: Independent term in kernel function (0.0)
# Ouput:
    # SVM Model, Training Accuracy, Testing Accuracy
def runSVM(train_x, train_y, test_x, test_y, rC, kernel, degree, gamma, coef0):
    if (config['MODELS']['SVM'] == False):
        return None, None, None
    
    C = rC  # Regularization parameter

    svm = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, class_weight='balanced', probability=True)

    print("training SVM model...")
    svm.fit(train_x, train_y)
    print("done!")

    # Model Statistics
    print("getting statistics...")

    y_trainPred = svm.predict(train_x)
    trainStats = get_infield_statistics(train_y, y_trainPred)
    train_accuracy = trainStats[0]
    train_averageError = trainStats[1]

    y_pred = svm.predict(test_x)
    testStats = get_infield_statistics(test_y, y_pred)
    test_accuracy = testStats[0]
    test_averageError = testStats[1]

    train_stat = [train_accuracy, train_averageError]
    test_stat = [test_accuracy, test_averageError]
    
    logs.logModel("SVM", svm, train_stat, test_stat, [train_x, train_y, test_x, test_y, y_trainPred, y_pred],
                   ["Regularization Constant: ", rC, "Kernel Type: ", kernel, "Kernel Degree", degree, "Kernel Coefficient (gamma): ", gamma, "Independent Term in Kernel (coef0): ", coef0])
    
    logs.printModel("SVM", svm, train_stat, test_stat, [train_x, train_y, test_x, test_y, y_trainPred, y_pred],
                   ["Regularization Constant: ", rC, "Kernel Type: ", kernel, "Kernel Degree", degree, "Kernel Coefficient (gamma): ", gamma, "Independent Term in Kernel (coef0): ", coef0])

    print("done!")

    return svm, train_accuracy, test_accuracy

# Run Random Forest Regressor
# Inputs:
    # train_x
    # train_y
    # test_x
    # test_y
def runRFR(train_x, train_y, test_x, test_y):
    rfr = RandomForestRegressor()
    param_grid = {
        'n_estimators': [100, 200, 400, 800, 1200],
        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'criterion': ['squared_error','absolute_error','friedman_mse','poisson'],
        'min_samples_split': [2, 4, 8, 12],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    best_rfr = trainHyperParameters(rfr, param_grid, train_x, train_y)
    best_rfr.fit(train_x, train_y)

    #importances = rfr.feature_importances_
    predictions = best_rfr.predict(test_x)
    directionScore, distanceScore = measurePerformance(predictions, test_y)

    return directionScore, distanceScore

def trainHyperParameters(model, grid, train_x, train_y):
    grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(train_x, train_y)
    return grid_search.best_estimator_

def measurePerformance(predictions, test_y):
    # Values From DataUtil, if you change those: change these, and vise-versa
    ANGLE_RANGE      = 55
    DISTANCE_RANGE   = 450


    direction_bins   = [-float('inf'), -45, -27, -9, 9, 27, 45, float('inf')]

    predictDirection = (predictions[0][0]*ANGLE_RANGE*2) - ANGLE_RANGE
    actualDirection  = (test_y.iloc[0].values[0]*ANGLE_RANGE*2) - ANGLE_RANGE
    errorDirection   = abs(predictDirection - actualDirection)

    predictSlice     = convertAngleToSlice(predictDirection)
    actualSlice      = convertAngleToSlice(actualDirection)

    predictDistance  = predictions[0][1] * 450
    actualDistance   = test_y.iloc[0].values[1] * 450
    errorDistance    = abs(predictDistance - actualDistance)

    directionScore   = 1 - math.sqrt(errorDirection / (2*ANGLE_RANGE))
    distanceScore    = 1 - math.sqrt(errorDistance / DISTANCE_RANGE)

    #print("Direction:\nPredict: ", predictDirection," (", predictSlice,")\tActual: ", actualDirection, " (", actualSlice, "),\tError: ", errorDirection,"\tScore: ", directionScore)
    #print("Distance:\nPredict: ", predictDistance, "\tActual: ", actualDistance, "\tError: ", errorDistance, "\tScore: ",distanceScore,"\n")

    return directionScore, distanceScore

# Function for calculating the weighted score of a model
# Inputs:
    # y_true: actual values
    # y_pred: predicted values
def calculateScore(y_true, y_pred):
    directionWeight = 3
    distanceWeight = 1
    measurePerformance(y_pred, y_true)
    weightedScore = ((directionScore * directionWeight) + (distanceScore * distanceWeight)) / (directionWeight + distanceWeight)
    return weightedScore

def convertAngleToSlice(angle):
    if angle < -27:
        return 1
    elif angle < -9:
        return 2
    elif angle < 9:
        return 3
    elif angle < 27:
        return 4
    else:
        return 5

def colsum(arr, n, m):
    coll = [0,0,0,0,0]
    for i in range(n):
        su = 0;
        for j in range(m):
            su += arr[j][i]
        coll[i] = su/m
    return coll 


def measurePerformance(predictions, test_y):
    # Values From DataUtil, if you change those: change these, and vise-versa
    ANGLE_RANGE      = 55
    DISTANCE_RANGE   = 450


    direction_bins   = [-float('inf'), -45, -27, -9, 9, 27, 45, float('inf')]

    predictDirection = (predictions[0][0]*ANGLE_RANGE*2) - ANGLE_RANGE
    actualDirection  = (test_y.iloc[0].values[0]*ANGLE_RANGE*2) - ANGLE_RANGE
    errorDirection   = abs(predictDirection - actualDirection)

    predictSlice     = convertAngleToSlice(predictDirection)
    actualSlice      = convertAngleToSlice(actualDirection)

    predictDistance  = predictions[0][1] * 450
    actualDistance   = test_y.iloc[0].values[1] * 450
    errorDistance    = abs(predictDistance - actualDistance)

    directionScore   = 1 - math.sqrt(errorDirection / (2*ANGLE_RANGE))
    distanceScore    = 1 - math.sqrt(errorDistance / DISTANCE_RANGE)

    #print("Direction:\nPredict: ", predictDirection," (", predictSlice,")\tActual: ", actualDirection, " (", actualSlice, "),\tError: ", errorDirection,"\tScore: ", directionScore)
    #print("Distance:\nPredict: ", predictDistance, "\tActual: ", actualDistance, "\tError: ", errorDistance, "\tScore: ",distanceScore,"\n")

    return directionScore, distanceScore

# Function for calculating the weighted score of a model
# Inputs:
    # y_true: actual values
    # y_pred: predicted values
def calculateScore(y_true, y_pred):
    directionWeight = 3
    distanceWeight = 1
    measurePerformance(y_pred, y_true)
    weightedScore = ((directionScore * directionWeight) + (distanceScore * distanceWeight)) / (directionWeight + distanceWeight)
    return weightedScore
    

def convertAngleToSlice(angle):
    if angle < -27:
        return 1
    elif angle < -9:
        return 2
    elif angle < 9:
        return 3
    elif angle < 27:
        return 4
    else:
        return 5

# Function for getting statistics of an infield zone model
# Inputs:
    # pred: predicted values
    # y_test: actual values
# Output:
    # accuracy
def get_infield_statistics(pred, y_test):
    true1 = 0
    true2 = 0
    true3 = 0
    true4 = 0
    true5 = 0

    false1 = 0
    false2 = 0
    false3 = 0
    false4 = 0
    false5 = 0

    totalError = 0

    index = 0
    for i in pred:
        if i == 1:
            if y_test[index] == 1:
                true1 += 1
            else:
                false1 += 1
        if i == 2:
            if y_test[index] == 2:
                true2 += 1
            else:
                false2 += 1
        if i == 3:
            if y_test[index] == 3:
                true3 += 1
            else:
                false3 += 1
        if i == 4:
            if y_test[index] == 4:
                true4 += 1
            else:
                false4 += 1
        if i == 5:
            if y_test[index] == 5:
                true5 += 1
            else:
                false5 += 1
        
        error = abs(i - y_test[index])
        totalError += error

        index += 1

        totalTrue = true1 + true2 + true3 + true4 + true5
        accuracy = totalTrue / len(y_test)

        averageError = totalError / len(y_test)


    return accuracy, averageError

