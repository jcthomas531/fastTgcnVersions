import trimesh
import pickle
import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import trimeshExtractFaceLabels as tefl
import trimeshToDf_labels as ttdl
import trimeshToDfNoLabels as ttdnl
import dfToPlyExport as dtpe


#testing
inPly = "K:/teeth3DS/scanData/upperPly_cS/MHDYIWUS_U_cS.ply"
inMat = "K:/testDir/rotTest.pkl"
outPath = "K:/testDir/rotApply.ply"
labs = True

#pull variables from snakemake
inPly = sys.argv[1]
inMat = sys.argv[2]
outPath = sys.argv[3]
#sys.argv only accepts strings so the True passed will be "True", converting
from distutils.util import strtobool
labs = bool(strtobool(sys.argv[4])) #outter bool() may not be necessary, legacy

#load in mesh
mesh = trimesh.load(inPly, process = False)

#load in rotation matrix
with open(inMat, "rb") as f:
    mat = pickle.load(f)

#when the mesh is labeled, extract face color information, otherwise nothing
if labs == True:
    colorDf = tefl.trimeshExtractFaceLabels(mesh)

#apply transformation
mesh.apply_transform(mat)

#export
if labs == True:
    vertDf, faceDf = ttdl.trimeshToDf_labels(mesh, colorDf = colorDf)
elif labs == False:
    vertDf, faceDf = ttdnl.trimeshToDfNoLabels(mesh)
else:
    raise ValueError("must have a parameter indicating if mesh has labels")


dtpe.dfToPlyExport(vertDf = vertDf, faceDf = faceDf, outFile = outPath)

