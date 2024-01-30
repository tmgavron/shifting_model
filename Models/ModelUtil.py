from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
# import modelanalysis as ma

from Logs import logging as logs

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
    dt = DecisionTreeClassifier(max_depth=max_depth, max_features=max_features, max_leaf_nodes=max_leaf_nodes, class_weight='balanced')
    #dt = DecisionTreeClassifier()   
    # Train Model
    print("training decision tree model...")
    dt.fit(train_x, train_y)
    print("done!")

    # Model Statistics
    print("getting statistics...\n")

    y_trainPred = dt.predict(train_x)
    train_accuracy = get_infield_statistics(train_y, y_trainPred)

    y_pred = dt.predict(test_x)
    test_accuracy = get_infield_statistics(test_y, y_pred)

    # Filtering for Statistics
    
    dftrain = train_x.copy()
    dftrain["FieldSlicePrediction"] = y_trainPred #add to columns
    dftrain["FieldSliceActual"] = train_y
    dftrain = dftrain.assign(Correct = lambda x: (x["FieldSliceActual"] == x["FieldSlicePrediction"]))
    
    #print(dftrain.groupby(["Correct"]).size())

    dftest = test_x.copy()
    dftest["FieldSlicePrediction"] = y_pred #add to columns
    dftest["FieldSliceActual"] = test_y
    dftest = dftest.assign(Correct = lambda x: (x["FieldSliceActual"] == x["FieldSlicePrediction"]))
    

    dfall = pd.concat([dftrain, dftest]) # add rows

    #print(dfall.groupby(["FieldSliceActual"]).size())
    #print(dfall.groupby(["FieldSlicePrediction"]).size())
    #print(dfall.groupby(["Correct"]).size())

    # Can either leave this code below or essentially switch it all with var=dftrain["FieldSliceActual"].value_counts()[1]
    dfTestStats = dftest.groupby(["FieldSliceActual"]).size().reset_index()
    dfTestStats = dfTestStats.rename(columns={"FieldSliceActual":"Field Slice",0:"Count of Actual"})
    dfTestStats["Count of Predicted"] = dftest.groupby(["FieldSlicePrediction"]).size().reset_index()[0]
    dftemp = dftest[dftest["Correct"] == True]
    dfTestStats["Correct"] = dftemp.groupby(["FieldSliceActual"]).size().reset_index()[0]

    dfTrainStats = dftrain.groupby(["FieldSliceActual"]).size().reset_index()
    dfTrainStats = dfTrainStats.rename(columns={"FieldSliceActual":"Field Slice",0:"Count of Actual"})
    dfTrainStats["Count of Predicted"] = dftrain.groupby(["FieldSlicePrediction"]).size().reset_index()[0]
    dftemp = dftrain[dftrain["Correct"] == True]
    dfTrainStats["Correct"] = dftemp.groupby(["FieldSliceActual"]).size().reset_index()[0]

    print("Decisions Tree Data Splits: train=0.75, test=0.25")
    print("Decision Tree Depth: " + str(dt.get_depth()))
    print("Decision Tree Number of Leaves: " + str(dt.get_n_leaves()))
    print("Decision Tree's Accuracy Score for Predicting on Training Data: " + str('{:.4f}'.format(train_accuracy)))
    print("Decision Tree's Accuracy Score for Predicting on Test Data: " + str('{:.4f}'.format(test_accuracy)))
    probs = dt.predict_proba(test_x)
    colprob = colsum(probs, len(probs[0]), len(probs))
    colperc = ['{:.2f}'.format(n*100) for n in colprob]
    print("\nDecision Tree Overall Average Probabilities\n-------------------------------------" )
    print("Section 1: " + str(colperc[0]) + "%\nSection 2: " + str(colperc[1]) + "%\nSection 3: " + str(colperc[2]) + "%")
    print("Section 4: " + str(colperc[3]) + "%\nSection 5: " + str(colperc[4]) + "%")
    print("Decision Tree Field Slice Counts for Training Data\n--------------------------------------------------")
    print("Section\tTruth\tPrediction")
    for i in range(dfTrainStats["Field Slice"].size):
        print(str(dfTrainStats["Field Slice"][i]) +"\t\t"+ str(dfTrainStats["Count of Actual"][i]) +"\t\t"+ str(dfTrainStats["Count of Predicted"][i]))
    print("Amount Correct: " + str(dftrain["Correct"].value_counts()[True]))
    print("Amount Incorrect: " + str(dftrain["Correct"].value_counts()[False]))
    print("Decision Tree Field Slice Counts for Testing Data\n--------------------------------------------------")
    print("Section\tTruth\tPrediction")
    for i in range(dfTestStats["Field Slice"].size):
        print(str(dfTestStats["Field Slice"][i]) +"\t\t"+ str(dfTestStats["Count of Actual"][i]) +"\t\t"+ str(dfTestStats["Count of Predicted"][i]))
    print("Amount Correct: " + str(dftest["Correct"].value_counts()[True]))
    print("Amount Incorrect: " + str(dftest["Correct"].value_counts()[False]))

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
    nb = GaussianNB(var_smoothing=var_smoothing, class_weight='balanced')
    
    # Train Model
    print("training Naive Bayes model...")
    nb.fit(train_x, train_y)
    print("done!")

    # Model Statistics
    print("getting statistics...")

    y_trainPred = nb.predict(train_x)
    train_accuracy = get_infield_statistics(train_y, y_trainPred)

    y_pred = nb.predict(test_x)
    test_accuracy = get_infield_statistics(test_y, y_pred)

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
    logreg = LogisticRegression(C=lr, max_iter=e, class_weight='balanced')

    print("training logistic regression model...")
    logreg.fit(train_x, train_y)
    print("done!")

    # Model Statistics
    print("getting statistics...")

    y_trainPred = logreg.predict(train_x)
    train_accuracy = get_infield_statistics(train_y, y_trainPred)

    y_pred = logreg.predict(test_x)
    test_accuracy = get_infield_statistics(test_y, y_pred)

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
    C = rC  # Regularization parameter

    svm = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, class_weight='balanced')

    print("training SVM model...")
    svm.fit(train_x, train_y)
    print("done!")

    # Model Statistics
    print("getting statistics...")

    y_trainPred = svm.predict(train_x)
    train_accuracy = get_infield_statistics(train_y, y_trainPred)

    y_pred = svm.predict(test_x)
    test_accuracy = get_infield_statistics(test_y, y_pred)

    print("done!")

    return svm, train_accuracy, test_accuracy


def colsum(arr, n, m):
    coll = [0,0,0,0,0]
    for i in range(n):
        su = 0;
        for j in range(m):
            su += arr[j][i]
        coll[i] = su/m
    return coll 


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

        index += 1

        totalTrue = true1 + true2 + true3 + true4 + true5
        accuracy = totalTrue / len(y_test)
    return accuracy
