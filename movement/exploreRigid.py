import open3d as o3d
import pandas as pd
import numpy as np
import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import readAndFormat as raf
import getRegistration as gr


prePath = "K:/iowaExpTest/segResults/segResults_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/pre/pat001Pre_formCSOriMastRemesh_seg.ply"
postPath = "K:/iowaExpTest/segResults/segResults_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/post/pat001Post_formCSOriMastRemesh_seg.ply"

preDat = raf.readAndFormat(prePath, arch = "U")
postDat = raf.readAndFormat(postPath, arch = "U")

#function that takes the data as read in via readAndFormat and returns a similarly
#structured version that just restricts to tooth of interest
#dat is the data in the format from readAndFormat
#toothNum is "1" to "16" including "gum", this must be a character
def restrictMeshToTooth(dat, toothNum):
    
    #separate face and vert data
    vDat = dat["vert"]
    fDat = dat["face"]
    
    #restrict face data to only those with number of interest
    fDatRest = fDat.loc[fDat["toothNum"] == toothNum]
    
    #index of associated vertices
    vertIndRest = list(set(
        vert
        for sublist in fDatRest["vertex_indices"]
        for vert in sublist
        ))
    
    #restrict vertex data to only those associated with tooth of interest
    vDatRest = vDat.iloc[vertIndRest]
    
    return {"vert": vDatRest, "face": fDatRest}


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
    preToothDict[i] = restrictMeshToTooth(dat = preDat, toothNum = i)
    postToothDict[i] = restrictMeshToTooth(dat = postDat, toothNum = i)
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
for i in teethInBoth:
    #get registration
    regTransi = gr.getRegistration(source = postPCDicts[i], target = prePCDicts[i])
    #store transformation in dictionary
    trans[i] = regTransi.transformation










