import pickle
import sys

import numpy as np
import open3d as o3d
import pandas as pd

#sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
sys.path.append("tools")
import getRegistration as gr
import readAndFormat as raf
import restrictMeshToTooth as rmtt

#testing
# prePath = "K:/iowaExpTest/segResults/segResults_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/pre/pat001Pre_formCSOriMastRemesh_seg.ply"
# postPath = "K:/iowaExpTest/segResults/segResults_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/post/pat001Post_formCSOriMastRemesh_rugAnnotSuperimp_seg.ply"
# outPath = "K:/iowaExpTest/spatialTrans/pat001_test.pkl"
#get values from snakmake
prePath = sys.argv[1]
postPath = sys.argv[2]
outPath = sys.argv[3]

#load in data
preDat = raf.readAndFormat(prePath, arch = "U")
postDat = raf.readAndFormat(postPath, arch = "U")

#create dictionary of data restricted to individual teeth
#all tooth list
tooth = list(map(str, list(range(1,17))))
tooth.append("gum")

#data restricted to each tooth
preToothDict = {}
postToothDict = {}
#data frame for tracking if each tooth is present in the pre and post scan
toothPresFrame = pd.DataFrame({
    "toothNum": tooth,
    "pre": np.nan,
    "post": np.nan,
    "both": np.nan
    })

#loop thru teeth
for i in tooth:
    #dictionaries for each tooths data
    preToothDict[i] = rmtt.restrictMeshToTooth(dat = preDat, toothNum = i)
    postToothDict[i] = rmtt.restrictMeshToTooth(dat = postDat, toothNum = i)
    #tracking tooth presence
    #is the tooth in the pre scan?
    if preToothDict[i]["vert"].shape[0] == 0:
        toothPresFrame.loc[toothPresFrame["toothNum"] == i, "pre"] = 0
    else: 
        toothPresFrame.loc[toothPresFrame["toothNum"] == i, "pre"] = 1 
    #is the tooth present in the post scan
    if postToothDict[i]["vert"].shape[0] == 0:
        toothPresFrame.loc[toothPresFrame["toothNum"] == i, "post"] = 0
    else:
        toothPresFrame.loc[toothPresFrame["toothNum"] == i, "post"] = 1
    #is the thooth present in both scans
    if (toothPresFrame.loc[toothPresFrame["toothNum"] == i, "pre"].iloc[0] == 1) & (toothPresFrame.loc[toothPresFrame["toothNum"] == i, "post"].iloc[0] == 1):
        toothPresFrame.loc[toothPresFrame["toothNum"] == i, "both"] = 1
    else:
        toothPresFrame.loc[toothPresFrame["toothNum"] == i, "both"] = 0

#dictionaries for point clouds for teeth present in both pre and post
teethInBoth = list(toothPresFrame.loc[toothPresFrame["both"] == 1]["toothNum"])
#point clouds for each tooth
prePCDicts = {}
postPCDicts = {}

#loop thru teeth in both
for i in teethInBoth:
    #format vertex data
    preVDati = preToothDict[i]["vert"].loc[:,["x", "y", "z"]].to_numpy()
    postVDati = postToothDict[i]["vert"].loc[:,["x", "y", "z"]].to_numpy()
    #convert to pointcloud
    prePCi = o3d.geometry.PointCloud()
    prePCi.points = o3d.utility.Vector3dVector(preVDati)
    postPCi = o3d.geometry.PointCloud()
    postPCi.points = o3d.utility.Vector3dVector(postVDati)
    #store point clouds in dictionaries
    prePCDicts[i] = prePCi
    postPCDicts[i] = postPCi
#o3d.visualization.draw_geometries([prePCDicts["2"], postPCDicts["2"]])


#get rigid transformation for each tooth
trans = {}

#loop thru teeth in both
#SWITCHING DIRECTION, (THINK ABOUT THEORY OF THIS, DOES IT MATTER WHAT IS WHAT IN ALGORITHM)
for i in teethInBoth:
    #get registration
    regTransi = gr.getRegistration(source = prePCDicts[i], target = postPCDicts[i])
    #store transformation in dictionary
    trans[i] = regTransi.transformation

#outputting
filePath = open(outPath, "wb")  # noqa: SIM115
pickle.dump(obj = trans,
            file = filePath)
filePath.close()
#can be read in like: 
# with open(outPath, "rb") as f:
#     obj = pickle.load(f)
