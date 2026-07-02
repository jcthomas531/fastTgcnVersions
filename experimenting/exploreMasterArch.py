import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import numpy as np
import trimesh
import trimeshExtractFaceLabels as tefl
import math
import trimeshToDf_labels as ttdl
import dfToPlyExport as dtpe


#master arch 1, with artificial base

#load in mesh
m1Path = "K:/teeth3DS/scanData/upperPly/0AAQ6BO3_U.ply"
m1Mesh = trimesh.load(m1Path, process = False)
#extract labels for later use
colorDf = tefl.trimeshExtractFaceLabels(m1Mesh)

#center mesh
m1Mesh.apply_translation(-m1Mesh.centroid)
#scale mesh to 1
scaleFac = 1/np.max(m1Mesh.extents)
m1Mesh.apply_scale(scaleFac)

#rotations to orient in desired manner
#https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html
#first rotation: around z axis
#set up components
radZ = math.radians(177)
sinZ = math.sin(radZ)
cosZ = math.cos(radZ)
#build matrix by row
r1Z = [cosZ, -sinZ, 0, 0]
r2Z = [sinZ, cosZ, 0, 0]
r3Z = [0, 0, 1, 0]
r4Z = [0, 0, 0, 1]
matZ = np.array([r1Z, r2Z, r3Z, r4Z])
#apply transformation
m1Mesh.apply_transform(matZ)

#second rotation: around X axis
#set up components
radX = math.radians(9)
sinX = math.sin(radX)
cosX = math.cos(radX)
#build matrix by row
r1X = [1, 0, 0, 0]
r2X = [0, cosX, sinX, 0]
r3X = [0, -sinX, cosX, 0]
r4X = [0, 0, 0, 1]
matX = np.array([r1X, r2X, r3X, r4X])
#apply transfromation
m1Mesh.apply_transform(matX)

#format and export
m1Vert, m1Face = ttdl.trimeshToDf_labels(m1Mesh, colorDf=colorDf)
#dtpe.dfToPlyExport(vertDf =  m1Vert, faceDf = m1Face, outFile = "K:/testDir/mA1Full.ply")
#import readAndPlot as rap
#rap.readAndPlot(file = "K:/testDir/mA1Full.ply", arch = "U")


