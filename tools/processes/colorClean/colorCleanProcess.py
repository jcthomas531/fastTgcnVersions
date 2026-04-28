import os
os.chdir("H:\\schoolFiles\\dissertation\\intraoralSegmentation\\tools")
import plyFunctions as pf
os.chdir("H:\\schoolFiles\\dissertation\\intraoralSegmentation\\tools\\colorClean")
import colorCleanFuns as cc
trainDir = "P:\\cph\\BIO\\Faculty\\gown\\research\\ThesisProjects\\Thomas\\IOSSegData\\original\\train"
testDir = "P:\\cph\\BIO\\Faculty\\gown\\research\\ThesisProjects\\Thomas\\IOSSegData\\original\\test"
import pandas as pd


# step 1, find which files need to have color cleaning
#for training data
os.chdir(trainDir)
trainDiscrip = pd.DataFrame([cc.numExtract(i) for i in os.listdir(trainDir)],
             columns=['fileName', 'pat', 'arch', 'numTeeth', 'verts', 'faces', 'anyNaTeeth'])
trainNas = trainDiscrip[trainDiscrip["anyNaTeeth"] == True]

#for test data
os.chdir(testDir)
testDiscrip =  pd.DataFrame([cc.numExtract(i) for i in os.listdir(testDir)],
             columns=['fileName', 'pat', 'arch', 'numTeeth', 'verts', 'faces', 'anyNaTeeth'])
testNas = testDiscrip[testDiscrip["anyNaTeeth"] == True]

# step 2, clean all of those files

def readAndClean(fileName, arch):
    dat = pf.readAndFormat(fileName, arch)
    datClean = cc.colorCleaner(dat)
    return datClean

#dont want to figure out how to do an apply with multiple args rn so using this
def cleanApplyer(dfNas):
    #create dicionary to hold objects
    outHolder = {}
    for i in range(len(dfNas)):
        #name dictionary element and give it the cleaned data
        outHolder[dfNas["fileName"].iloc[i]] = readAndClean(
            dfNas["fileName"].iloc[i],
            dfNas["arch"].iloc[i]
            )
    return outHolder

os.chdir(trainDir)
cleanTrain = cleanApplyer(trainNas)
os.chdir(testDir)
cleanTest = cleanApplyer(testNas)

# step 3, check all of these changes

# #for training data
# trainNas
# os.chdir(trainDir)
# fileString = "087_L.ply"
# trainDat = pf.readAndFormat(fileString, arch = "L")
# #examine issue
# cc.plotIssue(trainDat)
# #ensure it is changes to correct color
# pf.plotPly(face = cleanTrain[fileString]["face"], vertex = cleanTrain[fileString]["vert"])
# #28L good
# #35L good
# #36L good
# #38L good
# #45L good
# #45U good
# #46L good
# #53L good
# #65U good
# #71U good
# #87L good

# #for test data
# testNas
# os.chdir(testDir)
# fileString = "006_L.ply"
# testDat = pf.readAndFormat(fileString, arch = "L")
# #examine issue
# cc.plotIssue(testDat)
# #ensure it is changes to correct color
# pf.plotPly(face = cleanTest[fileString]["face"], vertex = cleanTest[fileString]["vert"])
# #06L good
# #27L good, biggest change but changed correctly

#functions that test for errors in the plotting and cleaning functions on the 
#cleaned data, we are looking for these to throw back errors as this means there is 
#nothing more to fix
def plotError(dat):
    try: 
        cc.plotIssue(dat)
    except ValueError:
        return "proper response"
    return "improper response, still color issues"


def cleanError(dat):
    try:
        cc.colorCleaner(dat)
    except Exception:
        return "proper response"
    return "improper response, still color issues"

plotError(list(cleanTrain.values())[1])
len(cleanTrain.values())
len(cleanTrain)
list(cleanTrain.keys())[1]



#checking train data
trainCheck = pd.DataFrame({
    "fileName": [pd.NA]*len(cleanTrain),
    "plotResponse": [pd.NA]*len(cleanTrain),
    "cleanResponse": [pd.NA]*len(cleanTrain)
    })
for i in range(len(trainNas)):
    trainCheck["fileName"].loc[i] = list(cleanTrain.keys())[i]
    trainCheck["plotResponse"].loc[i] =plotError(list(cleanTrain.values())[i])
    trainCheck["cleanResponse"].loc[i] = cleanError(list(cleanTrain.values())[i])
   
    
#checking the test data
testCheck = pd.DataFrame({
    "fileName": [pd.NA]*len(cleanTest),
    "plotResponse": [pd.NA]*len(cleanTest),
    "cleanResponse": [pd.NA]*len(cleanTest)
    })
for i in range(len(testNas)):
    testCheck["fileName"].loc[i] = list(cleanTest.keys())[i]
    testCheck["plotResponse"].loc[i] =plotError(list(cleanTest.values())[i])
    testCheck["cleanResponse"].loc[i] = cleanError(list(cleanTest.values())[i])
    

trainCheck
testCheck

#perfect, everything is how it should be


# step 3, export all of those files

#testing with test 06L
# test06LForm = cc.faceFormatter(list(cleanTest.values())[0])
# os.chdir("H:\\schoolFiles\\dissertation\\intraoralSegmentation\\testDir")
# cc.writePly("006Test_L.ply", test06LForm)
# cc.fullExport("006Test2_L.ply", test06LForm)
# #this seems to work and can be read in an operated on with previous functions
# plyTest1 = pf.readAndFormat("006Test2_L.ply", "L")
# pf.plotPly(face = plyTest1["face"], vertex = plyTest1["vert"])
# cc.numExtract("006Test2_L.ply")




cleanTrainDir = "P:\\cph\\BIO\\Faculty\\gown\\research\\ThesisProjects\\Thomas\\IOSSegData\\clean\\trainClean"
cleanTestDir = "P:\\cph\\BIO\\Faculty\\gown\\research\\ThesisProjects\\Thomas\\IOSSegData\\clean\\testClean"



#applying cc.fullExport to each element of the cleanTrain and cleanTest dictionaries
#applying via dictionary comprehension
#this says create a new dictionary where the indice is k and the value is f(v) for
#all of the value pairs k, v in the dictionary
#for training data
os.chdir(cleanTrainDir)
{k: cc.fullExport(k, v) for k, v in cleanTrain.items()}
#for test data
os.chdir(cleanTestDir)
{k: cc.fullExport(k, v) for k, v in cleanTest.items()}

# step 4, move all of the other files over
from pathlib import Path
import shutil
#for training data
trainNoNas = trainDiscrip[trainDiscrip["anyNaTeeth"] == False]

origDir = Path("P:\\cph\\BIO\\Faculty\\gown\\research\\ThesisProjects\\Thomas\\IOSSegData\\original\\train")
newDir = Path("P:\\cph\\BIO\\Faculty\\gown\\research\\ThesisProjects\\Thomas\\IOSSegData\\clean\\trainClean")

for i in trainNoNas["fileName"]:
    origFile = origDir / i
    newFile = newDir / i
    shutil.copy2(origFile, newFile)
    print(i + "moved to new directory")

#for test data
testNoNas = testDiscrip[testDiscrip["anyNaTeeth"] == False]

origDirTest = Path("P:\\cph\\BIO\\Faculty\\gown\\research\\ThesisProjects\\Thomas\\IOSSegData\\original\\test")
newDirTest = Path("P:\\cph\\BIO\\Faculty\\gown\\research\\ThesisProjects\\Thomas\\IOSSegData\\clean\\testClean")

for i in testNoNas["fileName"]:
    origFile = origDirTest / i
    newFile = newDirTest / i
    shutil.copy2(origFile, newFile)
    print(i + "moved to new directory")
