# Imports
from Models import ModelUtil
from Data import Preprocessing, DataUtil
from Logs import logging as logs

import importlib
import configparser
import pickle

config = configparser.ConfigParser()
config.read('Data//config.ini')

importlib.reload(Preprocessing)
importlib.reload(ModelUtil)
importlib.reload(logs)

import warnings
warnings.filterwarnings("ignore")

infieldDataFrame = []
outfieldDataFrame = []
models = []

def loadData():
    # 1) Load all data from preprocessing 
    newprocessing = 'True' in config['DATA']['USE_NEW_PREPROCESSING']
    infieldDataFrame, outfieldDataFrame = Preprocessing.dataFiltering([], newprocessing)
    return infieldDataFrame, outfieldDataFrame


# Function to train all models based on settings from config
def trainModels(infieldDataFrame, outfieldDataFrame):
    models = {}
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
    if ("False" in config['TRAIN']['Testing']):
        runCount = 1
        print("Not Testing")
    for j in range(runCount):
            xTrain, xTest, yTrain, yTest = ModelUtil.modelDataSplitting(infieldDataFrame, j, 0.25,'InfieldTrainingFilter')

            if("True" in config['MODELS']['DTC']):
                dtOutput = ModelUtil.runDT(xTrain, yTrain, xTest, yTest, max_depth, max_features, max_leaf_nodes)
                models["DT"] = dtOutput
                if ("True" in config['DATA']['Pickle']):
                    # Save the model to a file
                    with open('Models/DecisionTree.pkl', 'wb') as file:
                        pickle.dump(dtOutput, file)

            if("True" in config['MODELS']['NB']):   
                nbOutput = ModelUtil.runNB(xTrain, yTrain, xTest, yTest, var_smoothing)
                models["NB"] = nbOutput
                if ("True" in config['DATA']['Pickle']):
                    # Save the model to a file
                    with open('Models/NaiveBayes.pkl', 'wb') as file:
                        pickle.dump(nbOutput, file)

            if("True" in config['MODELS']['LR']):
                logRegOutput = ModelUtil.runLogReg(xTrain, yTrain, xTest, yTest, lr, e)
                models["LR"] = logRegOutput
                if ("True" in config['DATA']['Pickle']):
                    # Save the model to a file
                    with open('Models/LogRegression.pkl', 'wb') as file:
                        pickle.dump(logRegOutput, file)

            if("True" in config['MODELS']['SVM']):
                svmOutput = ModelUtil.runSVM(xTrain, yTrain, xTest, yTest, rC, kernel, degree, gamma, coef0)
                models["SVM"] = svmOutput
                if ("True" in config['DATA']['Pickle']):
                    # Save the model to a file
                    with open('Models/SVM.pkl', 'wb') as file:
                        pickle.dump(svmOutput, file)

            # if("True" in config['MODELS']['RF']):
            #     for i in range(0, len(trainIn)):
            #         direction, distance = ModelUtil.runRFR(trainIn[i], trainOut[i], testIn[i], testOut[i])
    return models

# Function to load models from their pickle files
def loadModels():
    models = {}
    # Load the models from the files
    if("True" in config['MODELS']['DTC']):
        with open('Models/DecisionTree.pkl', 'rb') as file:
            dt = pickle.load(file)
            models["DT"] = dt

    if("True" in config['MODELS']['NB']):   
        with open('Models/NaiveBayes.pkl', 'rb') as file:
            nb = pickle.load(file)
            models["NB"] = nb

    if("True" in config['MODELS']['LR']):
        with open('Models/LogRegression.pkl', 'rb') as file:
            lr = pickle.load(file)
            models["LR"] = lr

    if("True" in config['MODELS']['SVM']):
        with open('Models/SVM.pkl', 'rb') as file:
            svm = pickle.load(file)
            models["SVM"] = svm
    
    return models

# Function to output all average pitcher photos from 'Data/PitchMetricAverages_AsOf_2024-03-11.csv'
def outputPitcherAverages(data, pitchingAveragesDF, models):
    predictionKey = []
    predictions = []
    for index in range(data.shape[0]):
        if index != 0:
            averageProbs, error = predictSinglePitcherStat(data.iloc[index], models) 
            if error == True:
                print(pitchingAveragesDF.iloc[index]["Pitcher"])
            else:
                player = pitchingAveragesDF.iloc[index][0].replace(",", "_")
                pitch = pitchingAveragesDF.iloc[index]["TaggedPitchType"]
                batterSide = pitchingAveragesDF.iloc[index]["BatterSide"]
                team = pitchingAveragesDF.iloc[index]["PitcherTeam"]

                predictionKey.append([player,pitch,batterSide,team])
                predictions.append(averageProbs)

    # batch_image_to_excel.create_excel() 

    # predictions holds the model prediction outputs
    # predictionKey holds the player, pitch, and batter side information for the corresponding index in the predictions
    return predictionKey, predictions

def predictSinglePitcherStat(dataPoint, models):
    averageProbs= []
    modelTypeCount = 0
    averageProbs.append([0,0,0,0,0])
    error = False
    # For each selected model (config), add in the predicted probabilities
    if("True" in config['MODELS']['DTC']):
        dt = models["DT"][0]
        try:
            averageProbs += dt.predict_proba([dataPoint])[0]
            modelTypeCount += 1
        except:
            error = True

    if("True" in config['MODELS']['NB']):   
        nb = models["NB"][0]
        try:
            averageProbs += nb.predict_proba([dataPoint])[0]
            modelTypeCount += 1
        except:
            error = True

    if("True" in config['MODELS']['LR']):    
        logReg = models["LR"][0]
        try:
            averageProbs += logReg.predict_proba([dataPoint])[0]
            modelTypeCount += 1
        except:
            error = True

    if("True" in config['MODELS']['SVM']):
        svm = models["SVM"][0]
        try:
            averageProbs += svm.predict_proba([dataPoint])[0]
            modelTypeCount += 1
        except:
            error = True

    # Average the selected model's probabilities 
    averageProbs = averageProbs / modelTypeCount

    return averageProbs, error


# Run this every monday:
# Load the data and the models (train models every monday)
infieldDataFrame, outfieldDataFrame = loadData() # replace CSV data with all current data (including new weekly data)
models = trainModels(infieldDataFrame, outfieldDataFrame) # re-train models on new data (based on config)
    # models = loadModels() # If you do not want to retrain the models run this instead of train

# Connect to database and pull pitcher averages from SQL
cur, conn = DataUtil.databaseConnect()
averagesData, pitchingAveragesDF = DataUtil.getPitcherAverages(cur, infieldDataFrame, outfieldDataFrame, "None")

# Run pitcher average predictions
predictionKey, predictions = outputPitcherAverages(averagesData, pitchingAveragesDF, models) # change this to output predictions to the sql database to be read in and visualized when opening that players page

# write to defensive_shift_model_values view 
DataUtil.writePitcherAverages(cur, conn, predictionKey, predictions)

# Close the connection
cur.close()
conn.close()