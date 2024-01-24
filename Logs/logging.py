from datetime import datetime

def writeLog(log, name="", descriptor=".txt"):
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H%M%S")
    with open(f"logs/{name}{dt_string}{descriptor}", 'w') as file:
        for row in log:
            file.write(row)
            file.write("\n")

def logDT(model, train_acc, test_acc, train_limit, test_limit):
    weights = [w for w in model.w]
    # lr = model.lr
    # epochs = model.epochs

    log = list()

    log.append(f"Training Accuracy = {train_acc}")
    log.append(f"Testing Accuracy = {test_acc}")
    log.append("")
    log.append(f"Training Size = {train_limit}")
    log.append(f"Testing Size = {test_limit}")
    log.append("")
    # log.append(f"Learning Rate = {lr}")
    # log.append(f"Epochs = {epochs}")
    log.append("")
    log.append("Weights:")

    # log += [",".join([str(w) for w in class_w]) for class_w in weights]

    writeLog(log, "Decision Tree model ")

def logNB(model, train_acc, test_acc, train_limit, test_limit):
    weights = [w for w in model.w]
    # lr = model.lr
    # epochs = model.epochs

    log = list()

    log.append(f"Training Accuracy = {train_acc}")
    log.append(f"Testing Accuracy = {test_acc}")
    log.append("")
    log.append(f"Training Size = {train_limit}")
    log.append(f"Testing Size = {test_limit}")
    log.append("")
    # log.append(f"Learning Rate = {lr}")
    # log.append(f"Epochs = {epochs}")
    log.append("")
    log.append("Weights:")

    # log += [",".join([str(w) for w in class_w]) for class_w in weights]

    writeLog(log, "Naive Bayes model ")

def logLR(model, train_acc, test_acc, train_limit, test_limit):
    weights = [w for w in model.w]
    lr = model.lr
    epochs = model.epochs

    log = list()

    log.append(f"Training Accuracy = {train_acc}")
    log.append(f"Testing Accuracy = {test_acc}")
    log.append("")
    log.append(f"Training Size = {train_limit}")
    log.append(f"Testing Size = {test_limit}")
    log.append("")
    log.append(f"Learning Rate = {lr}")
    log.append(f"Epochs = {epochs}")
    log.append("")
    log.append("Weights:")

    log += [",".join([str(w) for w in class_w]) for class_w in weights]

    writeLog(log, "LR model ")
    

def logSVM(model, train_acc, test_acc, train_limit, test_limit):
    C = .5  # Regularization parameter
    kernel = 'linear'  # Kernel type (can be 'linear', 'poly', 'rbf', 'sigmoid')
    degree = 3  # Degree of the polynomial kernel function (ignored by all other kernels)
    gamma = 'scale'  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
    coef0 = 0.0  # Independent term in kernel function

    log = list()

    log.append(f"Training Accuracy = {train_acc}")
    log.append(f"Testing Accuracy = {test_acc}")
    log.append("")
    log.append(f"Training Size = {train_limit}")
    log.append(f"Testing Size = {test_limit}")
    log.append("")
    log.append(f"Regularization Constant = {C}")
    log.append(f"Kernel Type = {kernel}")
    log.append("")

    writeLog(log, "SVM model ")
    
def logMLPModel(model, init_lr, hidden_layer_sizes, train_acc, test_acc, train_limit, test_limit):
    log = list()

    log.append(f"Inital Learning Rate = {init_lr}")
    log.append(f"Hidden layer sizes = [{','.join([str(num) for num in hidden_layer_sizes])}]")
    log.append("")

    log.append("Validation scores over iterations:")
    log += [str(score) for score in model.validation_scores_]
    log.append("")

    log.append(f"Training Accuracy = {train_acc}")
    log.append(f"Testing Accuracy = {test_acc}")
    log.append("")
    log.append(f"Training Size = {train_limit}")
    log.append(f"Testing Size = {test_limit}")
    log.append("")

    writeLog(log, "MLP model ")