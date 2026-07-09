import trimesh
import pyvista as pv
import json
import numpy as np
import pandas as pd
import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import colorNumFrame as cnf
import trimeshToDf_labels as ttdl

#function that takes scan data formated as an obj file with labels stored by point
#in a json file and outputs the data formatted in dataframes in the way our process uses them
#this is primarily designed for the teeth3ds data but could be extended to other 
#data that are potentially formatted this way
#objFile is the path to the obj file
#jsonFile is the path to the json file
def objJsonToDataFrames(objFile, jsonFile):
    
    #load in obj file
    meshTri = trimesh.load_mesh(objFile, process = False)

    #convert to pyvista object
    meshObj = pv.wrap(meshTri)

    #load in json file
    with open(jsonFile) as fp:
        labelDat = json.load(fp)
        
    #extarct important pieces for use later
    labelList = labelDat["labels"]
    labelArray = np.array(labelList)

    #give the pyvista object the appropriate labels
    meshObj.point_data["labels"] = labelList


    #majority rules implimentation
    #set up
    faces = meshObj.faces.reshape(-1, 4)[:, 1:]
    vertexLabsArray = labelArray[faces]
    #apply rule
    rng = np.random.default_rng(826)
    overallLabHolder = []
    for i in range(len(vertexLabsArray)):
        labsi = vertexLabsArray[i]
        uniqLabCount = len(set(labsi))
        #impliment majority rules rationale
        #this will work when there is only 1 or 2 label choices
        if uniqLabCount in [1,2]:
            overallLabHolder.append(pd.Series(labsi).value_counts().idxmax().astype(int))
        elif uniqLabCount == 3:
            #randomly select one of the three labels 
            overallLabHolder.append(rng.choice(labsi, 1, replace = False)[0].astype(int)) #selecting first object here so it is single dimesnional
        else:
            raise ValueError("unique label counts not 1, 2, or 3")

    #get color mapping data frame
    colorRefDef = cnf.colorNumFrame("U")

    #function thing for mapping each label number to a color
    #chatgpt wrote this piece, not exactly use what it is doing
    labelToRgba = {
        row["fdiNum"]: [row["red"], row["green"], row["blue"], 255]
        for _, row in colorRefDef.iterrows()
    }

    #perform mapping
    rgba = np.array([labelToRgba[l] for l in overallLabHolder], dtype=np.uint8)

    #make array into data frame
    colorDf = pd.DataFrame({"red": rgba[:, 0],
                            "green": rgba[:, 1],
                            "blue": rgba[:, 2],
                            "alpha": rgba[:, 3]})

    #convert pyvista object to trimesh
    meshObjTri = pv.to_trimesh(meshObj)

    #convert trimesh to data frames
    vertDf, faceDf = ttdl.trimeshToDf_labels(x = meshObjTri, colorDf = colorDf)
    return vertDf, faceDf


#example
# vertDf, faceDf = objJsonToDataFrames(objFile = "K:/teeth3DS/scanData/upper/019TUUZD/019TUUZD_upper.obj",
#                           jsonFile = "K:/teeth3DS/scanData/upper/019TUUZD/019TUUZD_upper.json")
# import plotArch
# plotArch.plotArch(face = faceDf, vertex = vertDf)





