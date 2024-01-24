import numpy as np
import matplotlib.pyplot as matplt
import math
import csv
import random

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
        listOfCols = ["PitchNo", "PitchUID", "PitcherThrows", "BatterSide", "TaggedPitchType", "AutoPitchType", "PitchCall", "TaggedHitType", "PlayResult", 
                         "RelSpeed", "VertRelAngle", "HorzRelAngle", "SpinRate", "SpinAxis", "InducedVertBreak", "PlateLocHeight", "PlateLocSide",
                         "ZoneSpeed", "VertApprAngle", "HorzApprAngle", "ExitSpeed", "Angle", "HitSpinRate", "PositionAt110X", "PositionAt110Y",
                         "PositionAt110Z", "Distance", "Direction", "Bearing"]
        for colName in listOfCols:
            indexDic[colName] = find_column_index(filename, colName)

        for row in csv_reader:
            raw_row = list()
            # ID's:
            raw_row.append(str(row[indexDic["PitchNo"]])) # PitchNo
            raw_row.append(str(row[indexDic["PitchUID"]])) # PitchUID
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
            raw_row.append(safe_float_conversion(row[indexDic["Bearing"]])) # Bearing (for outfield pop flys etc)

            # Add Datapoint
            raw_data.append(raw_row)
    return raw_data

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