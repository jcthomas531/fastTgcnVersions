import sys

import numpy as np
import open3d as o3d
import pandas as pd
import pyvista as pv

sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import getRegistration as gr
import readAndFormat as raf

prePath = "K:/iowaExpTest/segResults/segResults_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/pre/pat013Pre_formCSOriMastRemesh_seg.ply"
postPath = "K:/iowaExpTest/segResults/segResults_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/post/pat013Post_formCSOriMastRemesh_rugAnnotSuperimp_seg.ply"

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
    fDatRest = fDat.loc[fDat["toothNum"] == toothNum].copy()
    
    #index of associated vertices
    vertIndRest = [
        vert
        for sublist in fDatRest["vertex_indices"]
        for vert in sublist
        ]
    #only unique values, no duplicate vertices
    #this would also work with list(set(vertIndRest)), it just orders things a bit differently
    vertIndRest = list(dict.fromkeys(vertIndRest)) 
    #restrict vertex data to only those associated with tooth of interest
    vDatRest = vDat.iloc[vertIndRest].copy()





    #change vertex indices in face data to correspond to newly reset vertex ordering
    aaa = dict(zip(vDatRest.index, range(len(vDatRest.index))))
    fDatRest["vertex_indices"] = fDatRest["vertex_indices"].apply(
        lambda inds: [aaa[i] for i in inds]
        )
    
    #reset indices of both data frames
    fDatRest = fDatRest.reset_index(drop = True)
    vDatRest = vDatRest.reset_index(drop = True)
    

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




#can this be easily visualized
import readAndFormat as raf
import giveSurf
import toothCentroids as toCe

#load formatted meshes
preDf = raf.readAndFormat(file = prePath, arch = "U")
postDf = raf.readAndFormat(file = postPath, arch = "U")

#find centroids
preCent = toCe.toothCentroids(face = preDf["face"], vertex = preDf["vert"])
postCent = toCe.toothCentroids(face = postDf["face"], vertex = postDf["vert"])

#surfaces for pyvista
preSurf = giveSurf.giveSurf(face = preDf["face"], vertex = preDf["vert"])
postSurf = giveSurf.giveSurf(face = postDf["face"], vertex = postDf["vert"])

#apply translation from transformation matrix to the centroids
translation = pd.DataFrame.from_dict(
    {k: v[:3, 3] for k, v in trans.items()},
    orient="index",
    columns=["transX", "transY", "transZ"]
).reset_index(names="toothNum")

moveDf = translation.merge(preCent, on="toothNum", how = "left")
#THIS IS THE WRONG TRANSFORMATION, SEE TRANSFORM_POINT FOR CORRECT ONE
moveDf["newX"] = moveDf["x"] + moveDf["transX"]
moveDf["newY"] = moveDf["y"] + moveDf["transY"]
moveDf["newZ"] = moveDf["x"] + moveDf["transZ"]

oldPoints = moveDf[["x", "y", "z"]].to_numpy()
newPoints = moveDf[["newX", "newY", "newZ"]].to_numpy()

#begin plots and add points
# p1 = pv.Plotter()
# p1.add_mesh(preSurf, scalars = "rgba", rgb = True,  opacity = .6)
# p1.add_mesh(postSurf, scalars = "rgba", rgb = True,  opacity = .6)
# p1.add_points(oldPoints, render_points_as_spheres=True, point_size=10, color = "red")
# p1.add_points(newPoints, render_points_as_spheres=True, point_size=10, color = "green")
# p1.show()



def transform_point(row):
    T = trans[str(row["toothNum"])]  # use row["tooth"] directly if the keys are integers

    # Homogeneous point
    p = np.array([row["x"], row["y"], row["z"], 1.0])

    # Apply transformation
    p_new = T @ p

    return p_new[:3]



preCent2 = preCent[~preCent.toothNum.isin(["allScan", "noGum", "molar", "premolar", "posterior", "canine", "incisor", "anterior"])]
newPoints2 = preCent2.apply(transform_point, axis=1, result_type="expand")
newPoints2 = np.array(newPoints2)

p2 = pv.Plotter()
p2.add_mesh(preSurf, color = "white",  opacity = .9)
p2.add_mesh(postSurf, color = "green",  opacity = .6)
p2.add_points(oldPoints, render_points_as_spheres=True, point_size=10, color = "red")
p2.add_points(newPoints2, render_points_as_spheres=True, point_size=10, color = "green")
p2.show()
