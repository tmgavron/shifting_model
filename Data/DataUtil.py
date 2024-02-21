import numpy as np
import matplotlib.pyplot as matplt
import math
import csv
import random
import pandas as pd
import configparser
from ftplib import FTP
from pathlib import Path
from io import BytesIO

config = configparser.ConfigParser()
config.read('Data//config.ini')

# All the values we believe will be most important to the model
listOfCols = ["PitcherId", "BatterId", "PitcherThrows", "BatterSide", "TaggedPitchType", "AutoPitchType", "PitchCall", "TaggedHitType", "PlayResult", 
              "RelSpeed", "RelHeight", "RelSide", "VertRelAngle", "HorzRelAngle", "SpinRate", "SpinAxis", "InducedVertBreak", "VertBreak", "HorzBreak", "PlateLocHeight", "PlateLocSide",
              "ZoneSpeed", "VertApprAngle", "HorzApprAngle", "ExitSpeed", "Angle", "HitSpinRate", "PositionAt110X", "PositionAt110Y",
              "PositionAt110Z", "Distance", "Direction", "Bearing", "HitLaunchConfidence", "HitLandingConfidence"]

def getData():  
    df = pd.DataFrame()
    if("True" in config['DATA']['DB_API']):
        pass
    elif ("True" in config['DATA']['FTP_API']):
        df = getFTPData()
    elif ("True" in config['DATA']['RawData']):
        df = getRawDataFrame("Data/TrackMan_NoStuff_Master.csv")
    elif ("True" in config['DATA']['FileZillaCSV']):
        df = getRawDataFrame('Data/combined_dataset.csv')
    else:
        print("No Data Source Selected")
    return df


def getFTPData():
    dates = config['FTP']['EarliestMonth']+"-"+config['FTP']['EarliestDay']+"-"+config['FTP']['EarliestYear']+"_"+config['FTP']['LatestMonth']+"-"+config['FTP']['LatestDay']+"-"+config['FTP']['LatestYear']
    my_pitchfile = Path("./Data/PitchData/PitchData_"+dates+".pkl")
    my_positionfile = Path("./Data/PositionData/PositionData_"+dates+".pkl")
    if (my_pitchfile.exists() and my_positionfile.exists()):
        combinedPitchDF = pd.read_pickle("./Data/PitchData/PitchData_"+dates+".pkl")
        combinedPositionDF = pd.read_pickle("./Data/PositionData/PositionData_"+dates+".pkl")  
    else:
        ftp = FTP('ftp.trackmanbaseball.com')         # connect to host
        ftp.login(user='Auburn',passwd='kA#R2,KNAP')  # given user and passwd
        #ftp = FTP(host=config['FTP']['ServerName'])  # should be able to use these two lines instead of hardcoding but
        #ftp.login(user=config['FTP']['UserName'],passwd=config['FTP']['Password']) # it is not working for some reason 
        ftp.cwd('v3') # change into "v3" directory

        combinedPitchDF = pd.DataFrame()    # initialize dataframes
        combinedPositionDF = pd.DataFrame()

        yearslistall = ftp.nlst() # lists out files and folders in current directory
        # pairs down list so that it is within the constraints given in config file
        yearslist = [year for year in yearslistall if int(year) >= int(config['FTP']['EarliestYear']) and int(year) <= int(config['FTP']['LatestYear'])]
        print(yearslist)
        for year in yearslist:
            ftp.cwd(year)
            monthslistall = ftp.nlst()
            monthslist = monthslistall
            if (int(year) == int(config['FTP']['EarliestYear'])):
                monthslist = [month for month in monthslistall if int(month) >= int(config['FTP']['EarliestMonth'])]
            if (int(year) == int(config['FTP']['LatestYear'])):
                monthslist = [month for month in monthslistall if int(month) <= int(config['FTP']['LatestMonth'])]
            monthslistall = monthslist
            print(monthslistall)
            for month in monthslistall:
                ftp.cwd(month)
                dayslistall = ftp.nlst()
                dayslist = dayslistall
                if (int(month) == int(config['FTP']['EarliestMonth'])):
                    dayslist = [day for day in dayslistall if int(day) >= int(config['FTP']['EarliestDay'])]
                if (int(month) == int(config['FTP']['LatestMonth'])):
                    dayslist = [day for day in dayslistall if int(day) <= int(config['FTP']['LatestDay'])]
                dayslistall = dayslist
                print(dayslistall)
                for day in dayslistall:
                    ftp.cwd(day+'//CSV')
                    filelist = ftp.nlst()
                    temp = pd.DataFrame()
                    for file in filelist:
                        if '.csv' in file and 'position' not in file:
                            flo = BytesIO()
                            ftp.retrbinary('RETR ' + file, flo.write)
                            flo.seek(0)
                            print(file)
                            #temp = pd.read_fwf(flo)
                            try:
                                temp = pd.read_csv(flo) # read file into pandas df
                            except:
                                pass
                            print(temp.shape)
                            if (temp.shape[1] == 167 and temp.shape[0] > 0):
                                print("yes - 167")
                                if (combinedPitchDF.empty):
                                    combinedPitchDF = temp.copy()
                                else:
                                    combinedPitchDF = pd.concat([combinedPitchDF, temp], ignore_index=True)
                            else:
                                print("no")
                        elif '.csv' in file and 'position' in file:
                            ftp.retrbinary('RETR ' + file, flo.write) # retrieve binary data transferer
                            flo.seek(0) # goes back to start of file
                            print(file)
                            #temp = pd.read_fwf(flo)
                            try:
                                temp = pd.read_csv(flo) # read file into pandas df
                            except:
                                pass
                            print(temp.shape)
                            if (temp.shape[1] == 29 and temp.shape[0] > 0):
                                print("yes")
                                if (combinedPositionDF.empty):
                                    combinedPositionDF = temp.copy()
                                else:
                                    combinedPositionDF = pd.concat([combinedPositionDF, temp], ignore_index=True)
                            else:
                                print("no")
                    ftp.cwd('../../')
                ftp.cwd('../')
            ftp.cwd('../')
        ftp.quit()

        print("combinedPitchDF")
        #display(combinedPitchDF)
        print("combinedPositionDF")
        #display(combinedPositionDF)

        if ("True" in config['DATA']['Pickle']):
            combinedPitchDF.to_pickle("./Data/PitchData/PitchData_"+dates+".pkl")
            combinedPositionDF.to_pickle("./Data/PositionData/PositionData_"+dates+".pkl")
        # add logging for what is saved off

    return combinedPitchDF # combinedPositionDF not returned yet but can be used later on


# Reads the data from the given filename and returns it as a list in the following format:

# PitchNo   - Not needed
# PitchUID  - Cheating

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
def getRawDataFrame(filename):
    raw_data = list()
    with open(filename, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)

        # Skip the header row
        next(csv_reader)

        # Create List of column indexes from column name
        indexDic = {}
        for colName in listOfCols:
            indexDic[colName] = find_column_index(filename, colName)

        for row in csv_reader:
            raw_row = list()
            # ID's:
            #raw_row.append(str(row[indexDic["PitchNo"]])) # PitchNo
            #raw_row.append(str(row[indexDic["PitchUID"]])) # PitchUID
            raw_row.append(str(row[indexDic["PitcherId"]])) # PitcherId
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
            raw_row.append(safe_float_conversion(row[indexDic["RelHeight"]])) # RelHeight
            raw_row.append(safe_float_conversion(row[indexDic["RelSide"]])) # RelSide
            raw_row.append(safe_float_conversion(row[indexDic["VertRelAngle"]])) # VertRelAngle
            raw_row.append(safe_float_conversion(row[indexDic["HorzRelAngle"]])) # HorzRelAngle
            raw_row.append(safe_float_conversion(row[indexDic["SpinRate"]])) # SpinRate
            raw_row.append(safe_float_conversion(row[indexDic["SpinAxis"]])) # SpinAxis
            raw_row.append(safe_float_conversion(row[indexDic["InducedVertBreak"]])) # InducedVertBreak
            raw_row.append(safe_float_conversion(row[indexDic["VertBreak"]])) # VertBreak
            raw_row.append(safe_float_conversion(row[indexDic["HorzBreak"]])) # HorzBreak
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

    # Create dataframe
    raw_dataframe = pd.DataFrame(raw_data, columns=listOfCols)
    raw_dataframe.dropna(axis=0, how='any')        
    return raw_dataframe

def trimData(df):
    trimmed_df = df[listOfCols]
    return trimmed_df

# This function sets each column that should NOT behave as a numeric value to split columns with boolean values (0 or 1)
# This currently IGNORES pitcherID and batterID, even though they would be categorical. This is so the model can be trained on all pitchers and batters.
# Input:  the DataFrame
# Output: the transformed DataFrame
def convertStringsToValues(df):
    categorical_features = ["PitcherThrows","BatterSide","TaggedPitchType","AutoPitchType","PitchCall","TaggedHitType","PlayResult","HitLaunchConfidence","HitLandingConfidence","PitcherId","BatterId"]
    transformed_df = pd.get_dummies(df, columns=categorical_features, dtype=float)
    return transformed_df

# This function expunges all empty strings, bad datapoints, and NaN datapoints from the given DataFrame
# Input:  the DataFrame
# Output: the cleaned DataFrame
def expungeData(df):
    df.loc[(df['Direction']      > 55.00) | (df['Direction']      < -55.00), 'Direction']      = np.nan # Remove bad angles (direction)
    df.loc[(df['Bearing']        > 55.00) | (df['Bearing']        < -55.00), 'Bearing']        = np.nan # Remove bad angles (bearing)
    df.loc[(df['PlateLocSide']   >  1.75) | (df['PlateLocSide']   <  -1.75), 'PlateLocSide']   = np.nan # Remove bad pitches (horizontal)
    df.loc[(df['PlateLocHeight'] >  4.00) | (df['PlateLocHeight'] <   0.00), 'PlateLocHeight'] = np.nan # Remove bad pitches (vertical)
    df.loc[~df['PitchCall'].str.contains("InPlay"), 'PitchCall'] = np.nan                               # Remove bad hits

    df = df.replace('', np.nan)                                                                         # Remove empty Strings
    df = df.dropna(axis=0, how='any')                                                                   # Drop all NaN data points
    return df

# This function uses a custom normalization method (saturation) to normalize the data in the given DataFrame.
def normalizeData(df):
    normal_df = (df-df.min())/(df.max()-df.min())
    return normal_df

# This function filters the given Pandas DataFrame specifically for infield data fields. These fields are used just for initial testing and
#   training of the Models
# Inputs:
    # df: the fieldDataFrame
# Output: the filtered DataFrame
def infieldFilter(df):
    if("True" in config['DATA']['USE_NEW_PREPROCESSING']):
        # Setup headers
        pitch_hit    = df[['PitcherThrows_Right','PitcherThrows_Left','BatterSide_Right','BatterSide_Left']]
        tagged_pitch = df.filter(like='TaggedPitchType')
        tagged_hit   = df[['TaggedHitType_GroundBall']]
        pitch_info   = df[['ZoneSpeed','PlateLocHeight','PlateLocSide','VertApprAngle','HorzApprAngle','RelSpeed']]
        hit_info     = df[['Direction','Distance']]

        bins = [-45, -27, -9, 9, 27, 45]
        labels = [1,2,3,4,5]
        hit_info["FieldSlice"] = pd.cut(hit_info["Direction"], bins=bins, labels=labels)

        filtered_df = pd.concat([pitch_hit,tagged_pitch,tagged_hit,pitch_info], axis=1)
        filtered_x  = filtered_df.columns.tolist()
        filtered_df = pd.concat([filtered_df,hit_info], axis=1)

        # Filter rows that don't meet the criteria
        filtered_df = filtered_df[(filtered_df['TaggedHitType_GroundBall'] == 1) | (filtered_df['Distance'] <= 0.35)] # 0.35 is approx grassline
        return filtered_df, filtered_x

    else:
        # ----- PREVIOUS FILTERING -----
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
    # Setup headers
    if("True" in config['DATA']['USE_NEW_PREPROCESSING']):
        pitch_hit    = df[['PitcherThrows_Right','PitcherThrows_Left','BatterSide_Right','BatterSide_Left']]
        tagged_pitch = df.filter(like='TaggedPitchType')
        tagged_hit   = df[['TaggedHitType_FlyBall','TaggedHitType_LineDrive']]
        pitch_info   = df[['ZoneSpeed','PlateLocHeight','PlateLocSide','VertApprAngle','HorzApprAngle','RelSpeed']]
        hit_info     = df[['Direction','Distance']]

        filtered_df = pd.concat([pitch_hit,tagged_pitch,tagged_hit,pitch_info], axis=1)
        filtered_x  = filtered_df.columns.tolist()
        filtered_df = pd.concat([filtered_df,hit_info], axis=1)

        # Filter rows that don't meet the criteria
        filtered_df = filtered_df[((filtered_df['TaggedHitType_FlyBall'] == 1) | (filtered_df['TaggedHitType_LineDrive'] == 1)) & (filtered_df['Distance'] > 0.35)] # 0.35 is approx grassline
        return filtered_df, filtered_x
    else:
        # ----- PREVIOUS FILTERING -----
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
    
# This function normalizes the given DataFrame
# Input: DataFrame to normalize
# Output: normalized DataFrame
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def clamp01(df):
    result = df.copy()
    for feature_name in df.columns:
        result[feature_name] = df[feature_name].clip(0, 1)
    return result


    normal_df = (df-df.min())/(df.max()-df.min())
    return normal_df