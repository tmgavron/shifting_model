from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
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
    print("getting statistics...")

    y_trainPred = dt.predict(train_x)
    train_accuracy = get_infield_statistics(train_y, y_trainPred)

    y_pred = dt.predict(test_x)
    test_accuracy = get_infield_statistics(test_y, y_pred)

    #logs.writeLog()
    print(dt.get_depth())
    print(dt.get_n_leaves())
    print(dt.predict_proba(test_x)) # returns [0,0,1,0,0]
    print(accuracy_score(test_y,y_pred))

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
