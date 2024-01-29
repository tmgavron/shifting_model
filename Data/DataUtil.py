import numpy as np
import matplotlib.pyplot as matplt
import math
import csv
import random
import pandas as pd

# Reads the data from the given filename and returns it as a list in the following format:

# PitchNo
# PitchUID

# Setup Info:
    # PitcherThrows
    # BatterSide 
    # TaggedPitchType
    # AutoPitchType
    # PitchCall (look for in play)
    # TaggedHitType
    # PlayResult

# Pitch Stats:
    # RelSpeed
    # VertRelAngle
    # HorzRelAngle
    # SpinRate
    # SpinAxis
    # InducedVertBreak
    # PlateLocHeight
    # PlateLocSide
    # ZoneSpeed
    # VertApprAngle
    # HorzApprAngle

# Hit Stats (used for classifying): 
    # ExitSpeed
    # Angle
    # HitSpinRate
    # PositionAt110X
    # PositionAt110Y
    # PositionAt110Z
    # Distance

# Labels (outcomes):
    # Direction (for infield ground balls)
    # Bearing (for outfield pop flys etc)

# Input: filename (name of file or path to file)
# Ouput: list of datapoints with desired columns
def getRawData(filename):
    raw_data = list()
    with open(filename, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)

        # Skip the header row
        next(csv_reader)

        # Create List of column indexes from column name
        indexDic = {}
        listOfCols = ["PitchNo", "PitchUID", "PitcherId", "BatterId", "PitcherThrows", "BatterSide", "TaggedPitchType", "AutoPitchType", "PitchCall", "TaggedHitType", "PlayResult", 
                         "RelSpeed", "VertRelAngle", "HorzRelAngle", "SpinRate", "SpinAxis", "InducedVertBreak", "PlateLocHeight", "PlateLocSide",
                         "ZoneSpeed", "VertApprAngle", "HorzApprAngle", "ExitSpeed", "Angle", "HitSpinRate", "PositionAt110X", "PositionAt110Y",
                         "PositionAt110Z", "Distance", "Direction", "Bearing", "HitLaunchConfidence", "HitLandingConfidence"]
        for colName in listOfCols:
            indexDic[colName] = find_column_index(filename, colName)

        for row in csv_reader:
            raw_row = list()
            # ID's:
            raw_row.append(str(row[indexDic["PitchNo"]])) # PitchNo
            raw_row.append(str(row[indexDic["PitchUID"]])) # PitchUID
            raw_row.append(str(row[indexDic["PitcherId"]])) # PitchId
            raw_row.append(str(row[indexDic["BatterId"]])) # BatterId
            # Setup Info:
            raw_row.append(str(row[indexDic["PitcherThrows"]])) # PitcherThrows
            raw_row.append(str(row[indexDic["BatterSide"]])) # BatterSide
            raw_row.append(str(row[indexDic["TaggedPitchType"]])) # TaggedPitchType
            raw_row.append(str(row[indexDic["AutoPitchType"]])) # AutoPitchType
            raw_row.append(str(row[indexDic["PitchCall"]])) # PitchCall (look for in play)
            raw_row.append(str(row[indexDic["TaggedHitType"]])) # TaggedHitType
            raw_row.append(str(row[indexDic["PlayResult"]])) # PlayResult
            # Pitch Stats:
            raw_row.append(safe_float_conversion(row[indexDic["RelSpeed"]])) # RelSpeed
            raw_row.append(safe_float_conversion(row[indexDic["VertRelAngle"]])) # VertRelAngle
            raw_row.append(safe_float_conversion(row[indexDic["HorzRelAngle"]])) # HorzRelAngle
            raw_row.append(safe_float_conversion(row[indexDic["SpinRate"]])) # SpinRate
            raw_row.append(safe_float_conversion(row[indexDic["SpinAxis"]])) # SpinAxis
            raw_row.append(safe_float_conversion(row[indexDic["InducedVertBreak"]])) # InducedVertBreak
            raw_row.append(safe_float_conversion(row[indexDic["PlateLocHeight"]])) # PlateLocHeight
            raw_row.append(safe_float_conversion(row[indexDic["PlateLocSide"]])) # PlateLocSide
            raw_row.append(safe_float_conversion(row[indexDic["ZoneSpeed"]])) # ZoneSpeed
            raw_row.append(safe_float_conversion(row[indexDic["VertApprAngle"]])) # VertApprAngle
            raw_row.append(safe_float_conversion(row[indexDic["HorzApprAngle"]])) # HorzApprAngle
            # Hit Stats:
            raw_row.append(safe_float_conversion(row[indexDic["ExitSpeed"]])) # ExitSpeed
            raw_row.append(safe_float_conversion(row[indexDic["Angle"]])) # Angle
            raw_row.append(safe_float_conversion(row[indexDic["HitSpinRate"]])) # HitSpinRate
            raw_row.append(safe_float_conversion(row[indexDic["PositionAt110X"]])) # PositionAt110X
            raw_row.append(safe_float_conversion(row[indexDic["PositionAt110Y"]])) # PositionAt110Y
            raw_row.append(safe_float_conversion(row[indexDic["PositionAt110Z"]])) # PositionAt110Z
            raw_row.append(safe_float_conversion(row[indexDic["Distance"]])) # Distance
            # Labels:
            raw_row.append(safe_float_conversion(row[indexDic["Direction"]])) # Direction (for infield ground balls)
            raw_row.append(safe_float_conversion(row[indexDic["Bearing"]])) # Bearing (for outfield flys balls etc)
            # Confidence:
            raw_row.append(str(row[indexDic["HitLaunchConfidence"]])) # Confidence of Direction being right (for infield ground balls)
            raw_row.append(str(row[indexDic["HitLandingConfidence"]])) # Confidence of Bearing being right (for outfield fly balls etc)

            # Add Datapoint
            raw_data.append(raw_row)
    return raw_data

# This function converts the Raw Data(from API) into a Pandas DataFrame for easy filtering and manipulation. Also easy to use inside of ML models. 
# Inputs:
    # data_list: a list of data ideally received from getRawData()
# Output: the converted DataFrame
def convertRawToDataFrame(data_list):
    listOfCols = ["PitchNo", "PitchUID", "PitcherId", "BatterId", "PitcherThrows", "BatterSide", "TaggedPitchType", "AutoPitchType", "PitchCall", "TaggedHitType", "PlayResult", 
                         "RelSpeed", "VertRelAngle", "HorzRelAngle", "SpinRate", "SpinAxis", "InducedVertBreak", "PlateLocHeight", "PlateLocSide",
                         "ZoneSpeed", "VertApprAngle", "HorzApprAngle", "ExitSpeed", "Angle", "HitSpinRate", "PositionAt110X", "PositionAt110Y",
                         "PositionAt110Z", "Distance", "Direction", "Bearing", "HitLaunchConfidence", "HitLandingConfidence"]
    fieldDataFrame = pd.DataFrame(data_list, columns=listOfCols)
    return fieldDataFrame

# This function filters the given Pandas DataFrame specifically for infield data fields. These fields are used just for initial testing and
#   training of the Models
# Inputs:
    # df: the fieldDataFrame
# Output: the filtered DataFrame
def infieldFilter(df):
    # df = df[["PitcherId","BatterId","TaggedPitchType","PitchCall","TaggedHitType","Direction","HitLaunchConfidence"]]
    # ^^^ That one was from before decision to do All Hitters vs PitchType
    df = df[["PitcherThrows", "BatterSide", "TaggedPitchType", "PitchCall", "TaggedHitType", "ZoneSpeed", "PlateLocHeight", "PlateLocSide", "Direction"]]
    df = df[df["PitcherThrows"].isin(["Left", "Right", "Both"])] # 1, 2, 3 (can remove Both)
    df["PitcherThrows"] = df["PitcherThrows"].map({"Left":1, "Right":2, "Both":3})
    df = df[df["BatterSide"].isin(["Left","Right"])] # 1, 2
    df["BatterSide"] = df["BatterSide"].map({"Left":1, "Right":2})
    df = df[df["TaggedPitchType"].isin(["Fastball", "Sinker", "Cutter", "Curveball", "Slider", "Changeup", "Splitter", "Knuckleball"])] # 1,2,3,4,5,6,7,8
    df["TaggedPitchType"] = df["TaggedPitchType"].map({"Fastball":1, "Sinker":2, "Cutter":3, "Curveball":4, "Slider":5, "Changeup":6, "Splitter":7, "Knuckleball":8})
    df = df[df["PitchCall"].str.contains("InPlay")]
    df = df[df["TaggedHitType"].str.contains("GroundBall")]
    df = df[df["Direction"].between(-45, 45)]
    bins = [-45, -27, -9, 9, 27, 45]
    labels = [1,2,3,4,5]
    df["FieldSlice"] = pd.cut(df["Direction"], bins=bins, labels=labels)
    # df = df[df["HitLaunchConfidence"].isin(["Medium","High"])]
    # print("--")
    # print(df)
    # print("--")
    return df

# This function filters the given Pandas DataFrame specifically for outfield data fields. These fields are used just for initial testing and
#   training of the Models
# Inputs:
    # df: the fieldDataFrame
# Output: the filtered DataFrame
def outfieldFilter(df):
    # df = df[["PitcherId","BatterId","TaggedPitchType","PitchCall","TaggedHitType","Bearing","Distance","HitLandingConfidence"]]
    df = df[df["PitchCall"].str.contains("InPlay")]
    df = df[df["TaggedHitType"].isin(["FlyBall","LineDrive"])]
    df = df[df["Distance"] >= 150]
    df = df[df["Bearing"].between(-45, 45)]
    bins = [-45, -27, -9, 9, 27, 45]
    labels = [1,2,3,4,5]
    df['FieldSlice'] = pd.cut(df['Bearing'], bins=bins, labels=labels)
    # df = df[df["HitLandingConfidence"].isin(["Medium","High"])]
    return df


# This function finds the index of a given column in a dataset
# Inputs: 
    # csv_file_path: name of file (path if not in folder),
    # header_name: name of desired column to find index
# Output: the index of the column in that file/dataset (or -1 if not found)
def find_column_index(csv_file_path, header_name):
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Read the first line as the header
        if header_name in headers:
            return headers.index(header_name)
        else:
            return -1  # Return -1 or raise an error if the header is not found

# This function handles N/A datapoints or other errors when converting to a float
# Input: value to convert to float 
# Output: value converted to float
def safe_float_conversion(value):
    try:
        return float(value)
    except ValueError:
        return float('nan')  # or use None if you prefer