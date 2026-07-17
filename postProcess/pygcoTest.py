
hpcIos = "../"
hpcLss = "../../../../../Shared/gb_lss/Thomas/"


#this will need to be run on the hpc via the lorwyn_eclipsed contianer
import sys
# sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
# sys.path.append("Y:/dissModels/intraoralSegmentation/tools/gco_python")
sys.path.append(hpcIos + "tools")
sys.path.append(hpcIos + "tools/gco_python")








#we need edges, unaries, pairwise


# m2Path = "K:/iowaExpTest/segResults/segResults_t3dsIosseg_cSOriMastEpoch300/origForm_cSOriMastRemesh/post/predMat/pat014Post_formCSOriMastRemesh_predMat.pkl"
m2Path = hpcLss + "iowaExpTest/segResults/segResults_t3dsIosseg_cSOriMastEpoch300/origForm_cSOriMastRemesh/post/predMat/pat014Post_formCSOriMastRemesh_predMat.pkl"
import pickle
with open(m2Path, "rb") as f:
    m2 = pickle.load(f)
import torch
import numpy as np
#originally, values are logged, make into probabilities
probs = torch.exp(m2).detach().cpu().numpy()


#
#pulling in from meshsegnet
#
patch_prob_output = probs
num_classes = 17

round_factor = 100
patch_prob_output[patch_prob_output<1.0e-6] = 1.0e-6
# unaries
unaries = -round_factor * np.log10(patch_prob_output)
unaries = unaries.astype(np.int32)
unaries = unaries.reshape(-1, num_classes)


# parawise
pairwise = (1 - np.eye(num_classes, dtype=np.int32))





meshPath = hpcLss + "iowaExpTest/segResults/segResults_t3dsIosseg_cSOriMastEpoch300/origForm_cSOriMastRemesh/post/pat014Post_formCSOriMastRemesh_seg.ply"


import vedo
vMesh = vedo.load(meshPath)
vMesh.compute_normals()
normals = vMesh.celldata["Normals"].copy()
barycenters = vMesh.cell_centers().coordinates
cell_ids = np.asarray(vMesh.cells) #they use mesh_d.faces() but i believe this accomplishes the same thing
# vMesh.ncells
# normals.shape
# barycenters.shape
# cell_ids.shape[0]


#his creates the weighting scheme to determine how much adjacent cells should
#have the same label, based on angel between cells and cell center distance
lambda_c = 30 #this is smoothing strength, generally smaller values imply less change
edges = np.empty([1, 3], order='C')
for i_node in range(cell_ids.shape[0]): #they use cells.shape[0] which is just a value stating the number of cells, since we are not using that portion i have changed to this, which is the same value
    # Find neighbors
    nei = np.sum(np.isin(cell_ids, cell_ids[i_node, :]), axis=1)
    nei_id = np.where(nei==2)
    for i_nei in nei_id[0][:]:
        if i_node < i_nei:
            cos_theta = np.dot(normals[i_node, 0:3], normals[i_nei, 0:3])/np.linalg.norm(normals[i_node, 0:3])/np.linalg.norm(normals[i_nei, 0:3])
            if cos_theta >= 1.0:
                cos_theta = 0.9999
            theta = np.arccos(cos_theta)
            phi = np.linalg.norm(barycenters[i_node, :] - barycenters[i_nei, :])
            if theta > np.pi/2.0:
                edges = np.concatenate((edges, np.array([i_node, i_nei, -np.log10(theta/np.pi)*phi]).reshape(1, 3)), axis=0)
            else:
                beta = 1 + np.linalg.norm(np.dot(normals[i_node, 0:3], normals[i_nei, 0:3]))
                edges = np.concatenate((edges, np.array([i_node, i_nei, -beta*np.log10(theta/np.pi)*phi]).reshape(1, 3)), axis=0)
edges = np.delete(edges, 0, 0)
edges[:, 2] *= lambda_c*round_factor #scaling weights up
edges = edges.astype(np.int32)

import pygco
refine_labels = pygco.cut_from_graph(edges, unaries, pairwise)
refine_labels = refine_labels.reshape([-1, 1])

#output refine labels
outPath = hpcLss + "iowaExpTest/testDir/pygcoTest1.pkl"
filePath = open(outPath, "wb")
pickle.dump(obj = refine_labels,
            file = filePath)
filePath.close()

#

#format new labels
labPath = "K:/iowaExpTest/testDir/pygcoTest1.pkl"
with open(labPath, "rb") as f:
    refine_labels = pickle.load(f)
import pandas as pd
labDf = pd.DataFrame({"toothNum": refine_labels.flatten()})
#wires for labeling seem to be crossed somewhere, problem for later
labDf["toothNum"] = labDf["toothNum"].astype(str)
labDf["toothNum"] = labDf["toothNum"].replace("0", "gum")

#join to color frame
import colorNumFrame as cnf
colFrame = cnf.colorNumFrame("U")
colFrame = colFrame.drop(columns = ["fdiNum"])
labDf = pd.merge(labDf, colFrame, on = "toothNum", how = "left")
labDf["arch"] = "upper"
labDf["alpha"] = 255



#get vertex data from scan
import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import readAndFormat as raf
path14 = "K:/iowaExpTest/segResults/segResults_t3dsIosseg_cSOriMastEpoch300/origForm_cSOriMastRemesh/post/pat014Post_formCSOriMastRemesh_seg.ply"
meshDat = raf.readAndFormat(path14, arch = "U")
vertInfo = meshDat["face"][["vertex_indices"]]


#join new label d↓ata to vertex data
newFace = vertInfo.join(labDf, how = "left")

#reformat
newFace = newFace[["vertex_indices", "red", "green", "blue", "alpha", "color", "toothNum", "arch"]]

import plotArch as plAr
plAr.plotArch(face = meshDat["face"], vertex = meshDat["vert"])
plAr.plotArch(face = newFace, vertex = meshDat["vert"])
import readAndPlot as rap
rap.readAndPlot(file = "K:/iowaExpTest/segResults/segResults_t3dsIosseg_cSOriMastEpoch300/origForm_cSOriMastRemesh/post/pat004Post_formCSOriMastRemesh_seg.ply", arch = "U")
rap.readAndPlot(file = "K:/iowaExpTest/testDir/pat004NewLabs.ply", arch = "U")
