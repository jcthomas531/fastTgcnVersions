from copy import copy
import sys
sys.path.append("tools")
import decim as d
import trimeshExtractFaceLabels as tefl
import trimeshToDf_labels as ttdl
import pyvista as pv
import trimesh

from faceToPointLabel_2Color import faceToPointLabel_2Color
import numpy as np

#testing
inFile = "K:/iowaExpTest/scanData/rugAnnotForm_cSOriMast/pre/pat001Pre_formCSOriMast.ply"
nPoints = 20000
outFile = "K:/iowaExpTest/testDir/testDecimOut.ply"

#convert number of points to number of faces (approximate)
nFace = nPoints * 2

#read in file and convert to face labels
meshDat = faceToPointLabel_2Color(inFile = inFile)
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

















aaa = pd.DataFrame(meshDec.cell_data["RGBA"], columns = ["red", "green", "blue", "alpha"])

#convert to trimesh object
meshDec_tri = pv.to_trimesh(meshDec)

tefl.trimeshExtractFaceLabels(meshDec_tri)

#export
outDf = ttdl.trimeshToDf_labels(meshDec_tri)















