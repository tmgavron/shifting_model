# %%
# Model Training:
# 1) Load all data from preprocessing (training/test splits, etc)
# 2) Begin Training Models
    #  a) Decision Tree
    #  b) Naive Bayes
    #  c) Logistic Regression
    #  d) SVM
# 3) Testing Models
# 4) New Iterations

# %%
# Imports
from Models import ModelUtil
from Data import Preprocessing, DataUtil
from Visualization import VisualUtil, batch_image_to_excel
from Logs import logging as logs

import importlib
import configparser
import numpy as np

config = configparser.ConfigParser()
config.read('Data//config.ini')

importlib.reload(Preprocessing)
importlib.reload(ModelUtil)
importlib.reload(VisualUtil)
importlib.reload(batch_image_to_excel)
importlib.reload(logs)

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

# %%
# 1) Load all data from preprocessing 
importlib.reload(Preprocessing)
newprocessing = 'True' in config['DATA']['USE_NEW_PREPROCESSING']
infieldDataFrame, outfieldDataFrame = Preprocessing.dataFiltering([], newprocessing)

# %%
# All of this is mapping the strings to numbers for both infieldDataFrame and outfieldDataFrame so that the correlation matrix can be computed
# This can most likely be moved to a method in the logging.py file
infieldDF4Matrix = infieldDataFrame.copy()
outfieldDF4Matrix = outfieldDataFrame.copy()
strColumns = [] 
for cName in outfieldDF4Matrix.columns:
    if(str(outfieldDF4Matrix[cName].dtype) in 'object'):
        strColumns.append(cName)
rValueDict = {}
for cName in strColumns:
    i = 0
    infieldUniques = infieldDF4Matrix[cName].unique()
    for rValue in infieldUniques:
        rValueDict.update({rValue:i})
        i+=1
    infieldDF4Matrix[cName] = infieldDF4Matrix[cName].map(rValueDict)
    uniqueVals = [x for x in outfieldDF4Matrix[cName].unique() if x not in infieldUniques]
    for rValue in uniqueVals: 
        rValueDict.update({rValue:i})
        i+=1
    outfieldDF4Matrix[cName] = outfieldDF4Matrix[cName].map(rValueDict)
infieldDF4Matrix = infieldDF4Matrix.replace(np.nan, 0)
infieldDF4Matrix = infieldDF4Matrix.replace('', 0)
outfieldDF4Matrix = outfieldDF4Matrix.replace(np.nan, 0)
outfieldDF4Matrix = outfieldDF4Matrix.replace('', 0)

# Correlation does not imply causation.
# -1 means that the 2 variables have an inverse linear relationship: when X increases, Y decreases
# 0 means no linear correlation between X and Y
# 1 means that the 2 variables have a linear relationship: when X increases, Y increases too.
infieldcorrmatrix = infieldDF4Matrix.corr()
outfieldcorrmatrix = outfieldDF4Matrix.corr()
if (config['LOGGING']['Excel'] == 'True'):
    logs.writeToExcelSheet(infieldcorrmatrix, "Infield Correlation Matrix")
    logs.writeToExcelSheet(outfieldcorrmatrix, "Outfield Correlation Matrix")
if (config['LOGGING']['Debug'] == 'True'):
    print(infieldcorrmatrix)
    print(outfieldcorrmatrix)

# %%
importlib.reload(logs)
# 2) Trains all Models and exports all data to an Excel Sheet
max_depth = 50
max_features = 30
max_leaf_nodes = 150
# could also add ways to change it for these hyperparams below for other models
var_smoothing = 1e-9
lr = 0.8
e = 100
rC = 1
kernel='linear'
degree= 1
gamma= 'scale'
coef0= 0.0

runCount = int(config['TRAIN']['TimesRun'])
if (config['TRAIN']['Testing'] == False):
     runCount = 1
for j in range(runCount):
        xTrain, xTest, yTrain, yTest = ModelUtil.modelDataSplitting(infieldDataFrame, j, 0.25,'InfieldTrainingFilter')
        print(xTrain)
        if("True" in config['MODELS']['DTC']):
            dtOutput = ModelUtil.runDT(xTrain, yTrain, xTest, yTest, max_depth, max_features, max_leaf_nodes)
        if("True" in config['MODELS']['NB']):   
            nbOutput = ModelUtil.runNB(xTrain, yTrain, xTest, yTest, var_smoothing)
        if("True" in config['MODELS']['LR']):
            logRegOutput = ModelUtil.runLogReg(xTrain, yTrain, xTest, yTest, lr, e)
        if("True" in config['MODELS']['SVM']):
            svmOutput = ModelUtil.runSVM(xTrain, yTrain, xTest, yTest, rC, kernel, degree, gamma, coef0)
        if("True" in config['MODELS']['RF']):
            for i in range(0, len(trainIn)):
                direction, distance = ModelUtil.runRFR(trainIn[i], trainOut[i], testIn[i], testOut[i])
     

# %%
importlib.reload(ModelUtil)
importlib.reload(logs)
# a) Decision Tree
# Need to test these hyperparameters for best case
# Maybe make a way to superset these
max_depth =      [50, 40]
max_features =   [30, 20]
max_leaf_nodes = [150, 100]
hyperparamlist = []
# This just makes the permutations of the hyperparameters above. Lets you test on many hyperparams.
for n in range(len(max_depth)):
    for k in range(len(max_features)):
        for m in range(len(max_leaf_nodes)):
            hyperparamlist.append([max_depth[n], max_features[k], max_leaf_nodes[m]])
            
# for each permutation, it runs a certain amount of time that you specify in the config (30 rn bc of Dozier) and saves the outcome to an excel sheet
# requires to rerun the training set every time because otherwise will give you the same outcome every time
# Also proves that its the models ability, not the luck of the draw for the data
for lst in hyperparamlist:
    runCount = int(config['TRAIN']['TimesRun'])
    if (config['TRAIN']['Testing'] == False):
        runCount = 1
    for j in range(runCount):
        xTrain, xTest, yTrain, yTest = ModelUtil.modelDataSplitting(infieldDataFrame, j, 0.25,'InfieldTrainingFilter')
        dtOutput = ModelUtil.runDT(xTrain, yTrain, xTest, yTest, lst[0], lst[1], lst[2])


# %%
# TODO
# This is meant to take all the values from the 30 runs and average them and output them to another sheet of averages for different models
# Then will need to do this for all the models
# Can take this and put it into an excelAverages function
#prob rename this

# could move these column letter names and do something with that so not hardcoded
if("True" in config['LOGGING']['Excel']):
    sColumns = ['Training Accuracy', 'Testing Accuracy', 'Training Average Error', 'Testing Average Error', 'Training F1(micro)', 'Training F1(macro)', 'Training F1(weighted)', 
                'Testing F1(micro)', 'Testing F1(macro)', 'Testing F1(weighted)', 'Training AUC(macro)', 'Training AUC(weighted)', 'Testing AUC(macro)', 'Testing AUC(weighted)', 
                'Section 0 Probability', 'Section 1 Probability', 'Section 2 Probability', 'Section 3 Probability', 'Section 4 Probability']
    if("True" in config['MODELS']['DTC']):
        # columns in excel: I J K L W X Y Z AA AB AC AD AE AF AG AH AI AJ AK   
        sColumnsLetter = ['I','J','K','L','W','X','Y','Z','AA','AB','AC','AD','AE','AF','AG','AH','AI','AJ','AK']
        logs.excelAverages('DecisionTree',sColumns,sColumnsLetter)
    if("True" in config['MODELS']['NB']):
        sColumnsLetter = ['D','E','F','G','R','S','T','U','V','W','X','Y','Z','AA','AB','AC','AD','AE','AF']
        logs.excelAverages('NaiveBayes',sColumns,sColumnsLetter)
    if("True" in config['MODELS']['LR']):
        sColumnsLetter = ['E','F','G','H','S','T','U','V','W','X','Y','Z','AA','AB','AC','AD','AE','AF','AG']
        logs.excelAverages('LogisticRegression',sColumns,sColumnsLetter)
    if("True" in config['MODELS']['SVM']):
        sColumnsLetter = ['H','I','J','K','V','W','X','Y','Z','AA','AB','AC','AD','AE','AF','AG','AH','AI','AJ']
        logs.excelAverages('SVM',sColumns,sColumnsLetter)
    if("True" in config['MODELS']['RF']):
        logs.excelAverages('RandomForest',sColumns,sColumnsLetter)


# %%
importlib.reload(ModelUtil)
importlib.reload(logs)
# b) Naive Bayes

var_smoothing = 1e-9
runCount = int(config['TRAIN']['TimesRun'])
if (config['TRAIN']['Testing'] == False):
     runCount = 1
for j in range(runCount):
        xTrain, xTest, yTrain, yTest = ModelUtil.modelDataSplitting(infieldDataFrame, j, 0.25,'InfieldTrainingFilter')
        nbOutput = ModelUtil.runNB(xTrain, yTrain, xTest, yTest, var_smoothing)

# %%
importlib.reload(ModelUtil)
importlib.reload(logs)
# c)Logistic Regression
lr = 0.8
e = 100
logRegOutput = ModelUtil.runLogReg(xTrain, yTrain, xTest, yTest, lr, e)

# %%
importlib.reload(ModelUtil)
importlib.reload(logs)
# d) SVM
rC = 1
kernel='linear'
degree= 1
gamma= 'scale'
coef0= 0.0
svmOutput = ModelUtil.runSVM(xTrain, yTrain, xTest, yTest, rC, kernel, degree, gamma, coef0)

# %%
# z) RandomForestRegressor
for i in range(0, len(trainIn)):
    direction, distance = ModelUtil.runRFR(trainIn[i], trainOut[i], testIn[i], testOut[i])

# %%
# Change the value of index to look at different datapoints
importlib.reload(VisualUtil)
# 3) Model Testing:
dt = dtOutput[0]
nb = nbOutput[0]
logReg = logRegOutput[0]
# svm = svmOutput[0]

print("Testing Output: ")
# index of test value:
index = 4555
print(f"Actual Field Slice: \t\t{yTest.iloc[index]}")

print("\nDecision Tree:")
print(f"Predicted Field Slice: \t\t{dt.predict([xTest.iloc[index]])[0]}")
print(f"Field Slice Probabilities: \t{dt.predict_proba([xTest.iloc[index]])[0]}")

print("\nNaive Bayes:")
print(f"Predicted Field Slice: \t\t{nb.predict([xTest.iloc[index]])[0]}")
print(f"Field Slice Probabilities: \t{nb.predict_proba([xTest.iloc[index]])[0]}")

print("\nLogistic Regression:")
print(f"Predicted Field Slice: \t\t{logReg.predict([xTest.iloc[index]])[0]}")
print(f"Field Slice Probabilities: \t{logReg.predict_proba([xTest.iloc[index]])[0]}")

# print("\nSVM:")
# print(f"Predicted Field Slice: \t\t{svm.predict([xTest.iloc[index]])[0]}")
# print(f"Field Slice Probabilities: \t{svm.predict_proba([xTest.iloc[index]])[0]}")

averageProbs = dt.predict_proba([xTest.iloc[index]])[0] + nb.predict_proba([xTest.iloc[index]])[0] + logReg.predict_proba([xTest.iloc[index]])[0] # + svm.predict_proba([xTest.iloc[index]])[0]
averageProbs = averageProbs / 3 

print(f"\n\nAVG Prediction: \t\t{np.argmax(averageProbs)+1}")
print(f"Field Slice AVG Probabilities: \t{averageProbs}")

VisualUtil.visualizeData(averageProbs, [1], 'TestPic.png')

# %%
# 5) Data Visualization
importlib.reload(VisualUtil)

# Temporary method of getting percentages for testing purposes
infieldPercentages  = np.random.dirichlet(np.ones(4), size=1)[0]
outfieldPercentages = np.random.dirichlet(np.ones(2), size=1)[0]
outfieldCoordinates = np.random.uniform(low=[-45, 150], high=[45, 400], size=(30,2))

VisualUtil.visualizeData(infieldPercentages, outfieldCoordinates, "FieldTest")


# %%
# Average Pitcher Data Processing and Running
importlib.reload(Preprocessing)
importlib.reload(DataUtil)
importlib.reload(VisualUtil)
importlib.reload(batch_image_to_excel)


pitchingAveragesDF = DataUtil.getRawDataFrame('Data/PitchMetricAverages_AsOf_2024-03-11.csv')
# drop nan values from the used columns
specific_columns = ["PitcherThrows", "BatterSide", "TaggedPitchType", "RelSpeed", "InducedVertBreak", "HorzBreak", "RelHeight", "RelSide", "SpinAxis", "SpinRate", "VertApprAngle", "HorzApprAngle"] # pitcher averages
infieldDataFrame = infieldDataFrame[specific_columns] 
averagesX = pitchingAveragesDF[specific_columns] # pitcher averages
#averagesX = averagesX[["PitcherThrows", "BatterSide", "TaggedPitchType", "PlateLocHeight", "PlateLocSide", "ZoneSpeed", "RelSpeed", "SpinRate", "HorzBreak", "VertBreak"]]

averagesX["PitcherThrows"] = averagesX["PitcherThrows"].map({"Left":1, "Right":2, "Both":3})
averagesX["BatterSide"] = averagesX["BatterSide"].map({"Left":1, "Right":2})
averagesX["TaggedPitchType"] = averagesX["TaggedPitchType"].map({"Fastball": 1, "FourSeamFastBall":1, "Sinker":2, "TwoSeamFastBall":2, "Cutter":3, "Curveball":4, "Slider":5, "ChangeUp":6, "Splitter":7, "Knuckleball":8})

# normalize this based on min and maxes from training data
averagesX = DataUtil.normalizeData(averagesX, infieldDataFrame)

# Change the value of index to look at different datapoints
importlib.reload(VisualUtil)
# 3) Model Testing:
dt = dtOutput[0]
nb = nbOutput[0]
logReg = logRegOutput[0]
# svm = svmOutput[0]
for index in range(pitchingAveragesDF.shape[0]):
    print(index)
    averageProbs= []
    averageProbs = dt.predict_proba([averagesX.iloc[index]])[0] + nb.predict_proba([averagesX.iloc[index]])[0] + logReg.predict_proba([averagesX.iloc[index]])[0]
    averageProbs = averageProbs / 3 

    # print(f"\n\nAVG Prediction: \t\t{np.argmax(averageProbs)+1}")
    # print(f"Field Slice AVG Probabilities: \t{averageProbs}")
    fileName = pitchingAveragesDF.iloc[index][0].replace(",", "_").replace(" ", "") + "_" + pitchingAveragesDF.iloc[index]["TaggedPitchType"] + "_" + pitchingAveragesDF.iloc[index]["BatterSide"] + "Batter"
    VisualUtil.visualizeData(averageProbs, [1], fileName)   

batch_image_to_excel.create_excel()

# %%
# # This is for putting the right visuals on the correct excel sheets
# # For each player in the pitching averages, have a whole excel page for them
# import os
# importlib.reload(logs)
# print(pitchingAveragesDF)
# picList = []
# fileList = os.listdir("Visualization")
# for x in pitchingAveragesDF["Pitcher"].unique():
#     for y in fileList:
#         if x.replace(",", "_").replace(" ", "") in y:
#             picList.append(y)
#     logs.writeToImageExcelSheet(picList,x)
#     picList = []


