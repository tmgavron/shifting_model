from datetime import datetime
import pandas as pd

def writeLog(log, name="", descriptor=".txt"):
    now = datetime.now()
    dt_string = now.strftime(" %d-%m-%Y %H %M")
    with open(f"logs/{name}{dt_string}{descriptor}", 'w') as file:
        for row in log:
            file.write(row)
            file.write("\n")

def logModel(modelType, model, train_stats, test_stats, data, params):
    # weights = [w for w in model.w]
    
    # log.append("Weights:")

    # log += [",".join([str(w) for w in class_w]) for class_w in weights]

    log = list()
    log.append(f"Model Type: {modelType}")
    log.append("")
    log.append(f"Training Size = {len(data[0])}")
    log.append(f"Testing Size = {len(data[2])}")
    log.append("")
    log.append(f"Training Accuracy = {train_stats[0]}")
    log.append(f"Testing Accuracy = {test_stats[0]}")
    log.append("")
    log.append(f"Training Average Error = {train_stats[1]}")
    log.append(f"Testing Average Error = {test_stats[1]}")
    log.append("")
    log.append(f"Training Recall = {train_stats[2]}")
    log.append(f"Testing Recall = {test_stats[2]}")
    
    log.append("")
    log.append("Hyper-Parameters: \n")
    i = 0
    while i < len(params):
        log.append(f"{params[i]}{params[i+1]}")
        i += 2
    log.append("")

    # Filtering for Statistics
    
    train_x = data[0]
    train_y = data[1]
    test_x = data[2]
    test_y = data[3]
    y_trainPred = data[4]
    y_pred = data[5]

    dftrain = train_x.copy()
    dftrain["FieldSlicePrediction"] = y_trainPred #add to columns
    dftrain["FieldSliceActual"] = train_y
    dftrain = dftrain.assign(Correct = lambda x: (x["FieldSliceActual"] == x["FieldSlicePrediction"]))

    dftest = test_x.copy()
    dftest["FieldSlicePrediction"] = y_pred #add to columns
    dftest["FieldSliceActual"] = test_y
    dftest = dftest.assign(Correct = lambda x: (x["FieldSliceActual"] == x["FieldSlicePrediction"]))
    
    dfall = pd.concat([dftrain, dftest]) # add rows

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


    log.append("Accuracy Score for Predicting on Training Data: " + str('{:.4f}'.format(train_stats[0])))
    log.append("Accuracy Score for Predicting on Test Data: " + str('{:.4f}'.format(test_stats[0])))

    probs = model.predict_proba(test_x)
    colprob = colsum(probs, len(probs[0]), len(probs))
    colperc = ['{:.2f}'.format(n*100) for n in colprob]
    log.append("\nOverall Average Probabilities\n-------------------------------------" )
    log.append("Section 1: " + str(colperc[0]) + "%\nSection 2: " + str(colperc[1]) + "%\nSection 3: " + str(colperc[2]) + "%")
    log.append("Section 4: " + str(colperc[3]) + "%\nSection 5: " + str(colperc[4]) + "%")
    log.append("")

    log.append("Field Slice Counts for Training Data\n--------------------------------------------------")
    log.append("Section\tTruth\tPrediction")
    for i in range(dfTrainStats["Field Slice"].size):
        log.append(str(dfTrainStats["Field Slice"][i]) +"\t\t"+ str(dfTrainStats["Count of Actual"][i]) +"\t\t"+ str(dfTrainStats["Count of Predicted"][i]))
    log.append("Amount Correct: " + str(dftrain["Correct"].value_counts()[True]))
    log.append("Amount Incorrect: " + str(dftrain["Correct"].value_counts()[False]))    
    log.append("")

    log.append("Field Slice Counts for Testing Data\n--------------------------------------------------")
    log.append("Section\tTruth\tPrediction")
    for i in range(dfTestStats["Field Slice"].size):
        log.append(str(dfTestStats["Field Slice"][i]) +"\t\t"+ str(dfTestStats["Count of Actual"][i]) +"\t\t"+ str(dfTestStats["Count of Predicted"][i]))
    log.append("Amount Correct: " + str(dftest["Correct"].value_counts()[True]))
    log.append("Amount Incorrect: " + str(dftest["Correct"].value_counts()[False]))

    writeLog(log, modelType)



def printModel(modelType, model, train_stats, test_stats, data, params):
    # weights = [w for w in model.w]
    
    # log.append("Weights:")

    # log += [",".join([str(w) for w in class_w]) for class_w in weights]

    log = list()
    print(f"Model Type: {modelType}")
    print("")
    print(f"Training Size = {len(data[0])}")
    print(f"Testing Size = {len(data[2])}")
    print("")
    print(f"Training Accuracy = {train_stats[0]}")
    print(f"Testing Accuracy = {test_stats[0]}")
    print("")
    print(f"Training Average Error = {train_stats[1]}")
    print(f"Testing Average Error = {test_stats[1]}")    
    print("")
    print(f"Training Recall = {train_stats[2]}")
    print(f"Testing Recall = {test_stats[2]}")
    
    print("")
    print("Hyper-Parameters: \n")
    i = 0
    while i < len(params):
        print(f"{params[i]}{params[i+1]}")
        i += 2
    print("")

    # Filtering for Statistics
    
    train_x = data[0]
    train_y = data[1]
    test_x = data[2]
    test_y = data[3]
    y_trainPred = data[4]
    y_pred = data[5]

    dftrain = train_x.copy()
    dftrain["FieldSlicePrediction"] = y_trainPred #add to columns
    dftrain["FieldSliceActual"] = train_y
    dftrain = dftrain.assign(Correct = lambda x: (x["FieldSliceActual"] == x["FieldSlicePrediction"]))

    dftest = test_x.copy()
    dftest["FieldSlicePrediction"] = y_pred #add to columns
    dftest["FieldSliceActual"] = test_y
    dftest = dftest.assign(Correct = lambda x: (x["FieldSliceActual"] == x["FieldSlicePrediction"]))
    
    dfall = pd.concat([dftrain, dftest]) # add rows

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


    print("Accuracy Score for Predicting on Training Data: " + str('{:.4f}'.format(train_stats[1])))
    print("Accuracy Score for Predicting on Test Data: " + str('{:.4f}'.format(test_stats[0])))

    probs = model.predict_proba(test_x)
    colprob = colsum(probs, len(probs[0]), len(probs))
    colperc = ['{:.2f}'.format(n*100) for n in colprob]
    print("\nOverall Average Probabilities\n-------------------------------------" )
    print("Section 1: " + str(colperc[0]) + "%\nSection 2: " + str(colperc[1]) + "%\nSection 3: " + str(colperc[2]) + "%")
    print("Section 4: " + str(colperc[3]) + "%\nSection 5: " + str(colperc[4]) + "%")
    print("")

    print("Field Slice Counts for Training Data\n--------------------------------------------------")
    print("Section\tTruth\tPrediction")
    for i in range(dfTrainStats["Field Slice"].size):
        print(str(dfTrainStats["Field Slice"][i]) +"\t\t"+ str(dfTrainStats["Count of Actual"][i]) +"\t\t"+ str(dfTrainStats["Count of Predicted"][i]))
    print("Amount Correct: " + str(dftrain["Correct"].value_counts()[True]))
    print("Amount Incorrect: " + str(dftrain["Correct"].value_counts()[False]))    
    print("")

    print("Field Slice Counts for Testing Data\n--------------------------------------------------")
    print("Section\tTruth\tPrediction")
    for i in range(dfTestStats["Field Slice"].size):
        print(str(dfTestStats["Field Slice"][i]) +"\t\t"+ str(dfTestStats["Count of Actual"][i]) +"\t\t"+ str(dfTestStats["Count of Predicted"][i]))
    print("Amount Correct: " + str(dftest["Correct"].value_counts()[True]))
    print("Amount Incorrect: " + str(dftest["Correct"].value_counts()[False]))



def colsum(arr, n, m):
    coll = [0,0,0,0,0]
    for i in range(n):
        su = 0;
        for j in range(m):
            su += arr[j][i]
        coll[i] = su/m
    return coll 
