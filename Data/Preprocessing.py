# Here We Will Read And Process all of our Data
# 1) Read data from file (csv sample to start), then eventually from the "API", then later from the new database
# 2) Create input datasets from the data (filtered based on specifications for infield/outfield models)
    # Filter the data to only be relevant batted balls
    # Feature Engineering: add desired features (maybe an aggresiveness metric based on the count, situation, etc)
    # Labels: create labels (y outputs) for field slices based on direction, etc
# 3) Create training/testing splits 

# %%
# Imports
from Data import DataUtil
import importlib

# Download Data (from API/Database)


# %%

def dataFiltering():
    # 1) Read Data from file into DataFrame:
    importlib.reload(DataUtil)
    fieldDataFrame = DataUtil.getData()
    print("fieldDataFrame")
    print(fieldDataFrame.columns)
    print(fieldDataFrame)

    # 2) Create/Filter infield/outfield datasets from the data
    infieldDataFrame = DataUtil.infieldFilter(fieldDataFrame)
    #print(infieldDataFrame.head())
    #print(infieldDataFrame.shape)
    outfieldDataFrame = DataUtil.outfieldFilter(fieldDataFrame)
    #print(outfieldDataFrame.head())
    #print(outfieldDataFrame.shape)
    
    return infieldDataFrame, outfieldDataFrame

def dataProcessing():
    # 1) Read Data from file:
    importlib.reload(DataUtil)
    # small dataset CONFIG
    # rawData = DataUtil.getRawData("Data/TrackMan_NoStuff_Master.csv")
    # full dataset CONFIG
    fieldDataFrame = DataUtil.getData()

    # 3) Expunge bad data
    cleanDataFrame = DataUtil.expungeData(fieldDataFrame)
    #print("\nCleaned Data:")
    #display(cleanDataFrame) # easiest for US to read

    # 4) Convert from categorical to purely numerical data
    numericDataFrame = DataUtil.convertStringsToValues(cleanDataFrame)
    #display(numericDataFrame)

    # 5) Normalize the data
    normalizedDataFrame = DataUtil.normalizeData(numericDataFrame)
    #print("\nNormalized Data:")
    #display(normalizedDataFrame) # easiest for AI to read

    return normalizedDataFrame

def dataFiltering(df):

    infieldDataFrame, infieldX = DataUtil.infieldFilter(df)
    print("\nInfield Data: (No Pitcher / Batter IDs)")
    #display(infieldDataFrame)

    outfieldDataFrame, outfieldX = DataUtil.outfieldFilter(df)
    print("\nOutfield Data: (No Pitcher / Batter IDs)")
    #display(outfieldDataFrame)
    
    return (infieldDataFrame, infieldX), (outfieldDataFrame, outfieldX)

# %%
# Setup Frame for models:

# Basic Model Inputs: (normalize)"PitcherThrows", (normalize)"BatterSide", "TaggedPitchType", (rm)"PitchCall", (rm)"TaggedHitType", "ZoneSpeed", "PlateLocHeight", "PlateLocSide", (rm)"Direction"

# filter x data:

# create y list

def testTrainSplit():

    return 

# %%
# b) Feature Engineering: (create new features) (Sections 1-5 based on direction) 
#    Section 1:[-45, -27) Section 2:[-27, -9) Section 3:[-9, 9] Section 4:(9, 27] Section 5:(27 to 45]


# %%
# c) Create Labels (Output Variables)
# for infield take from direction
# Maybe just split the direction into 5 even intervals for the slices?

# for outfield take from heading 

# %%
# 3) Create Training/Test Splits


