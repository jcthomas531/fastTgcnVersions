import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import trimesh
import numpy as np
import trimeshExtractFaceLabels as tefl
import trimeshToDf_labels as ttdl
import trimeshToDfNoLabels as ttdnl
import dfToPlyExport as dtpe

#testing
# inPath = "K:/iowaRme/preDelivAndFinalScans/finalScanU/fullScans/pat001u_fin.ply"
# outPath = "K:/testDir/testCS_2.ply"
# labs = True

#pull variables from snakemake
inPath = sys.argv[1]
outPath = sys.argv[2]
#sys.argv only accepts strings so the True passed will be "True", converting
from distutils.util import strtobool
labs = bool(strtobool(sys.argv[3])) #outter bool() may not be necessary, legacy


#load in mesh
mesh = trimesh.load(inPath, process = False)

#when the mesh is labeled, extract face color information, otherwise nothing
if labs == True:
    colorDf = tefl.trimeshExtractFaceLabels(mesh)

#center mesh
mesh.apply_translation(-mesh.centroid)

#obtain scaling factor
scaleFac = 1/np.max(mesh.extents)
#scale mesh
mesh.apply_scale(scaleFac)


#export
if labs == True:
    vertDf, faceDf = ttdl.trimeshToDf_labels(mesh, colorDf = colorDf)
elif labs == False:
    vertDf, faceDf = ttdnl.trimeshToDfNoLabels(mesh)
else:
    raise ValueError("must have a parameter indicating if mesh has labels")
    

dtpe.dfToPlyExport(vertDf = vertDf, faceDf = faceDf, outFile = outPath)


