import sys

sys.path.append("tools")
import decim as d
import numpy as np
import pyvista as pv
from dfToPlyExport import dfToPlyExport
from faceToPointLabel_2Color import faceToPointLabel_2Color
from trimeshToDfNoLabels import trimeshToDfNoLabels
from vtk.util.numpy_support import vtk_to_numpy

#testing
# inFile = "K:/iowaExpTest/scanData/rugAnnotForm_cSOriMast/pre/pat001Pre_formCSOriMast.ply"
# nPoints = 20000
# outFile = "K:/iowaExpTest/testDir/testDecimOut.ply"

#bring in snakemake variables
inFile = sys.argv[1]
outFile = sys.argv[2]
#sys.argv only accepts strings so converting to numeric
nPoints = int(sys.argv[3])


#convert number of points to number of faces (approximate)
nFace = nPoints * 2

#read in file and convert to face labels
meshDat = faceToPointLabel_2Color(inFile = inFile, labelColor='255-000-127', nonlabelColor='255-255-255')
vDat = meshDat["vert"]
fDat = meshDat["face"]

#make into pyvista object
points = vDat[["x", "y", "z"]].to_numpy()
faces = np.hstack([
    np.full((len(fDat), 1), 3),
    np.vstack(fDat["vertex_indices"].to_numpy())
]).astype(np.int64).ravel()
meshPv = pv.PolyData(points, faces)
#add labels
meshPv.point_data["labels"] = vDat["label"].to_numpy()
#meshPv.plot() #note that you cannot plot before decimating bc it changes some attributes of the object

#decimate
meshDec = d.decim(x = meshPv, nFace = nFace)


#now we need to go back to face labels
#we dont strictly "need" to, but it facilitates easy of export and fits with the rest of the pipeline
#following procedure in ccRugaeAnnotVertToFaceLab.py
#first step is get vertex labels and face verts in the right form
vertLabs = vtk_to_numpy(meshDec.point_data["labels"])
vertLabs = vertLabs.reshape(-1, 1).astype(np.float32)
faceIndices = meshDec.faces.reshape(-1,4)[:,1:]


#loop through each face and assign face label based on if any vertex is classified as 1 (rugae)
#create empty list for face labels
faceLabs = []
for i in faceIndices:
    #extract the point labels for a face
    labs = vertLabs[i]
    #how many of the points are labeled as rugae
    countRugLabeled = np.sum(labs == 1)
    #impliment majority rules to map vertex labels to face labels
    if countRugLabeled >= 2:
        faceLabs.append(1)
    else:
        faceLabs.append(0)
#make face labels in an array
faceLabs = np.array(faceLabs)

#prepare for export in standard format
#make decimated mesh into a trimesh
meshDec_tri = pv.to_trimesh(meshDec)
#format mesh into data frames as if it had no labels
vDat, fDat = trimeshToDfNoLabels(meshDec_tri)
#change color for rugae labeled faces to black
fDat.loc[faceLabs == 1, ["red", "green", "blue"]] = [255,0,127]
#export
dfToPlyExport(vertDf = vDat, faceDf = fDat, outFile = outFile)
