import pandas as pd
import numpy as np
import os
import re
import sys

sys.path.append("tools")
import readAndFormat as raf
import toothCentroids as toCe
import teethToCenterDist as ttcd
import centroidSize as ceSi
import archLength as arLe
import centToCentDist as ctcd





#testing
#sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
#dir_ = "K:/iowaExpTest/segResults/segResults_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/pre/"
#outCent = "K:/iowaExpTest/testDir/testaaa.csv"


#take values from snakemake
dir_ = sys.argv[1] 
outCent = sys.argv[2]
outLength = sys.argv[3]

#create dictionary of patient names and files
filesAll = os.listdir(dir_)
files = [i for i in filesAll if i.endswith(".ply")]
pats = [re.search(pattern = r"^pat[0-9]{3}", string = i).group() for i in files]
pathDict = dict(zip(pats, files))



#set up lists of teeth
molars = ["1","2","3","14","15","16"]
premolars = ["4", "5", "12", "13"]
posterior = molars + premolars
canines = ["6", "11"]
incisors = ["7", "8", "9", "10"]
anterior = canines + incisors

#prepare centroid size data frame
centSize = pd.DataFrame(np.nan, index=range(len(files)),
                          columns=[
                              "patNum", "fullCs", "molarCs",
                              "premolarCs", "posteriorCs", "canineCs",
                              "incisorCs", "anteriorCs"
                              ])
centSize["patNum"] = list(pathDict.keys())

#prepare arch length data frame
archLengthDf = pd.DataFrame(np.nan, index=range(len(files)),
                            columns=["patNum", "archLength", "fullPerimeter"])
archLengthDf["patNum"] = list(pathDict.keys())


#set up for across palatte distances
distPairs = [
    ["1", "16"],
    ["2", "15"],
    ["3", "14"],
    ["4", "13"],
    ["5", "12"],
    ["6", "11"],
    ["7", "10"],
    ["8", "9"]
    ]

#loop thru files and calculate centriod size
for i in list(centSize["patNum"]):
    
    #read in data
    pathi = dir_ + pathDict[i]
    dati = raf.readAndFormat(file = pathi, arch = "U") 
    
    #get centroids
    tci = toCe.toothCentroids(face = dati["face"], vertex = dati["vert"])
    
    #get distance to center
    fullDisti =  ttcd.teethToCenterDist(tci, teethNums = "all", center = "noGum")
    molarDisti = ttcd.teethToCenterDist(tci, teethNums = molars, center = "molar")
    premolarDisti = ttcd.teethToCenterDist(tci, teethNums = premolars, center = "premolar")
    posteriorDisti = ttcd.teethToCenterDist(tci, teethNums = posterior, center = "posterior")
    canineDisti = ttcd.teethToCenterDist(tci, teethNums = canines, center = "canine")
    incisorDisti = ttcd.teethToCenterDist(tci, teethNums = incisors, center = "incisor")
    anteriorDisti = ttcd.teethToCenterDist(tci, teethNums = anterior, center = "anterior")
    
    #calculate centroid size
    fullCsi = ceSi.centriodSize(fullDisti)
    molarCsi = ceSi.centriodSize(molarDisti)
    premolarCsi = ceSi.centriodSize(premolarDisti)
    posteriorCsi = ceSi.centriodSize(posteriorDisti)
    canineCsi = ceSi.centriodSize(canineDisti)
    incisorCsi = ceSi.centriodSize(incisorDisti)
    anteriorCsi = ceSi.centriodSize(anteriorDisti)
    
    #calculate centroid size
    centSize.loc[centSize["patNum"] == i,
                 [
                     "fullCs", "molarCs",
                 "premolarCs", "posteriorCs", "canineCs",
                 "incisorCs", "anteriorCs"
                 ]
                 ] = [
                     fullCsi, molarCsi,
                     premolarCsi, posteriorCsi, canineCsi,
                     incisorCsi, anteriorCsi
                     ] 
    
    #calculate arch length and various other distances
    lengthsi = arLe.archLength(tci)
    
    #across palatte distances
    
    distHolder = {
        "1-16": np.nan,
        "2-15": np.nan,
        "3-14": np.nan,
        "4-13": np.nan,
        "5-12": np.nan,
        "6-11": np.nan,
        "7-10": np.nan,
        "8-9": np.nan
        }
    holderKeys = list(distHolder.keys())
    for j in range(0, len(distPairs)):
        distHolder[holderKeys[j]] = ctcd.centToCentDist(x = tci,
                                                        tooth1 = distPairs[j][0],
                                                        tooth2 = distPairs[j][1])
        
    
    archLengthDf.loc[archLengthDf["patNum"] == i,
                     ["archLength", "fullPerimeter"] + holderKeys] = [
                             lengthsi["archLength"], lengthsi["fullPerimeter"]
                             ] + list(distHolder.values())

#export
centSize.to_csv(outCent)
archLengthDf.to_csv(outLength)

