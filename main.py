# Imports
from Models import ModelUtil
from Data import Preprocessing, DataUtil
from Visualization import VisualUtil, batch_image_to_excel
from Logs import logging as logs

import importlib
import configparser
import numpy as np
import pickle

config = configparser.ConfigParser()
config.read('Data//config.ini')

importlib.reload(Preprocessing)
importlib.reload(ModelUtil)
importlib.reload(VisualUtil)
importlib.reload(batch_image_to_excel)
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


# Function to train all models based on settings from config
def trainModels():
    if (infieldDataFrame == []):
        loadData() # Need to do this so we can normalize
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
def outputPitcherAverages():
    if (models == []):
        models = loadModels()
    if (infieldDataFrame == []):
        loadData() # Need to do this so we can normalize

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
    dt = models["DT"][0]
    nb = models["NB"][0]
    logReg = models["LR"][0]
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



# Run this every monday:
loadData() # replace CSV data with all current data in weekly load
trainModels() # re-train models on new data (based on config)
outputPitcherAverages() # change this to output predictions to the sql database to be read in and visualized when opening that players page