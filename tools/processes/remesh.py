import random
import numpy as np
import trimesh
import pyvista as pv
import pyacvd
import pandas as pd

import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import trimeshExtractFaceLabels as tefl
import colorNumFrame as cnf
import trimeshToDf_labels as ttdl
import trimeshToDfNoLabels as ttdnl
import dfToPlyExport as dtpe

#seed setting
import os
os.environ["OMP_NUM_THREADS"] = "1"
seed = 826
random.seed(seed)
np.random.seed(seed)



#testing
# inPath = "K:/teeth3DS/scanData/upperPly_cSOriMast/00OMSZGW_U_cSOriMast.ply"
# outPath = "K:/testDir/remeshTest.ply"
# labs = True


#pull variables from snakemake
inPath = sys.argv[1]
outPath = sys.argv[2]
#sys.argv only accepts strings so the True passed will be "True", converting
from distutils.util import strtobool
labs = bool(strtobool(sys.argv[3])) #outter bool() may not be necessary, legacy

#get color mapping data frame
colorRefDef = cnf.colorNumFrame("U")


#load in mesh
meshTri = trimesh.load(inPath, process = False)
#if there are face labels
#extract face labels for later use
#remove label data from trimesh object for simple conversion to pyvista
if labs == True:
    #extract face labels
    colorDf = tefl.trimeshExtractFaceLabels(meshTri)
    #remove label data from trimesh
    meshTri.metadata.pop("_ply_raw", None)


#wrap trimesh as pyvista object
meshPv = pv.wrap(meshTri)


#remesh
meshIso = meshPv.acvd.remesh(8500, subdivide=3)

#make back into trimesh
meshIsoTri = pv.to_trimesh(meshIso)

#if there are face labels
#calculate new face centers for use in matching labels
#find closest original triangle
#subset colorDf with these indices
if labs == True:
    #calculate new face centers
    isoFaceCenters = meshIsoTri.triangles_center
    #find the closest triangle from the original trimesh
    closestPoints, distance, triangleId = meshTri.nearest.on_surface(isoFaceCenters)
    #create new colorDf by subsetting original colorDf to points corresponding to triangleId
    newColorDf = colorDf.iloc[triangleId].reset_index(drop=True)

#exporting
if labs == True:
    vertDf, faceDf = ttdl.trimeshToDf_labels(x = meshIsoTri, colorDf = newColorDf)
elif labs == False:
    vertDf, faceDf = ttdnl.trimeshToDfNoLabels(x = meshIsoTri)
else:
    raise ValueError("must have a parameter indicating if mesh has labels")


dtpe.dfToPlyExport(vertDf = vertDf, faceDf = faceDf, outFile = outPath)







