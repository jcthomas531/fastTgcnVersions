import sys


#main source that I am following
#https://github.com/Tai-Hsien/MeshSegNet/blob/master/step6_predict_with_post_processing_pygco.py


#testing
hpcIos = "../"
hpcLss = "../../../../../Shared/gb_lss/Thomas/"
sys.path.append(hpcIos + "tools")
sys.path.append(hpcIos + "tools/gco_python")
predMatPath = hpcLss + "iowaExpTest/segResults/segResults_t3dsIosseg_cSOriMastEpoch300/origForm_cSOriMastRemesh/post/predMat/pat004Post_formCSOriMastRemesh_predMat.pkl"
segMeshPath = hpcLss + "iowaExpTest/segResults/segResults_t3dsIosseg_cSOriMastEpoch300/origForm_cSOriMastRemesh/post/pat004Post_formCSOriMastRemesh_seg.ply"
outMeshPath = hpcLss + "iowaExpTest/testDir/pat004NewLabs.ply"


#bring in values from snakemake





###############
import pickle
import torch
import numpy as np
import vedo
import pygco
import pandas as pd
#
import colorNumFrame as cnf
import readAndFormat as raf


#lambda_c is smoothing strength, generally smaller values imply less change
def applyGraphCuts(predMatPath, segMeshPath, round_factor = 100, lambda_c = 30):
    
    #set up
    num_classes = 17
    
    #bring in predicted log probabilities matrix and format as probabities
    with open(predMatPath, "rb") as f:
        predMat = pickle.load(f)
    #format as probabiities
    patch_prob_output = torch.exp(predMat).detach().cpu().numpy() #might not need .cpu here as i have changed how the matrix was output
    #lower bound on probabilities
    patch_prob_output[patch_prob_output<1.0e-6] = 1.0e-6
    
    # calculate unaries
    unaries = -round_factor * np.log10(patch_prob_output)
    unaries = unaries.astype(np.int32)
    unaries = unaries.reshape(-1, num_classes)

    # pairwise matrix
    pairwise = (1 - np.eye(num_classes, dtype=np.int32))

    #load in segmented mesh (probably doesnt actally need to be the segmented one, could be input one but consistency)
    vMesh = vedo.load(segMeshPath)
    vMesh.compute_normals()
    normals = vMesh.celldata["Normals"].copy()
    barycenters = vMesh.cell_centers().coordinates
    cell_ids = np.asarray(vMesh.cells) #they use mesh_d.faces() but i believe this accomplishes the same thing
    
    #calculate edges
    #this also creates the weighting scheme to determine how much adjacent cells should
    #have the same label, based on angel between cells and cell center distance
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
    
    #apply graph cuts algorithm
    refine_labels = pygco.cut_from_graph(edges, unaries, pairwise)
    refine_labels = refine_labels.reshape([-1, 1])


    #format labels
    labDf = pd.DataFrame({"toothNum": refine_labels.flatten()})
    #wires for labeling seem to be crossed somewhere, problem for later
    labDf["toothNum"] = labDf["toothNum"].astype(str)
    labDf["toothNum"] = labDf["toothNum"].replace("0", "gum")


    #join to color frame
    colFrame = cnf.colorNumFrame("U")
    colFrame = colFrame.drop(columns = ["fdiNum"])
    labDf = pd.merge(labDf, colFrame, on = "toothNum", how = "left")
    labDf["arch"] = "upper"
    labDf["alpha"] = 255

    #get vertex data from scan
    meshDat = raf.readAndFormat(segMeshPath, arch = "U")
    vertInfo = meshDat["face"][["vertex_indices"]]

    #join new label data to vertex data
    newFace = vertInfo.join(labDf, how = "left")

    #reformat
    newFace = newFace[["vertex_indices", "red", "green", "blue", "alpha", "color", "toothNum", "arch"]]
    
    return meshDat["vert"], newFace

#test
vDat, fDat = applyGraphCuts(predMatPath = predMatPath, segMeshPath = segMeshPath)
import dfToPlyExport as dtpe
dtpe.dfToPlyExport(vertDf = vDat, faceDf = fDat, outFile = outMeshPath)



# #set up
# num_classes = 17
# round_factor = 100

# #bring in predicted log probabilities matrix and format as probabities
# with open(predMatPath, "rb") as f:
#     predMat = pickle.load(f)
# #format as probabiities
# patch_prob_output = torch.exp(predMat).detach().cpu().numpy() #might not need .cpu here as i have changed how the matrix was output
# #lower bound on probabilities
# patch_prob_output[patch_prob_output<1.0e-6] = 1.0e-6


# # calculate unaries
# unaries = -round_factor * np.log10(patch_prob_output)
# unaries = unaries.astype(np.int32)
# unaries = unaries.reshape(-1, num_classes)

# # pairwise matrix
# pairwise = (1 - np.eye(num_classes, dtype=np.int32))

# #load in segmented mesh (probably doesnt actally need to be the segmented one, could be input one but consistency)
# vMesh = vedo.load(segMeshPath)
# vMesh.compute_normals()
# normals = vMesh.celldata["Normals"].copy()
# barycenters = vMesh.cell_centers().coordinates
# cell_ids = np.asarray(vMesh.cells) #they use mesh_d.faces() but i believe this accomplishes the same thing

# #calculate edges
# #this also creates the weighting scheme to determine how much adjacent cells should
# #have the same label, based on angel between cells and cell center distance
# lambda_c = 30 #this is smoothing strength, generally smaller values imply less change
# edges = np.empty([1, 3], order='C')
# for i_node in range(cell_ids.shape[0]): #they use cells.shape[0] which is just a value stating the number of cells, since we are not using that portion i have changed to this, which is the same value
#     # Find neighbors
#     nei = np.sum(np.isin(cell_ids, cell_ids[i_node, :]), axis=1)
#     nei_id = np.where(nei==2)
#     for i_nei in nei_id[0][:]:
#         if i_node < i_nei:
#             cos_theta = np.dot(normals[i_node, 0:3], normals[i_nei, 0:3])/np.linalg.norm(normals[i_node, 0:3])/np.linalg.norm(normals[i_nei, 0:3])
#             if cos_theta >= 1.0:
#                 cos_theta = 0.9999
#             theta = np.arccos(cos_theta)
#             phi = np.linalg.norm(barycenters[i_node, :] - barycenters[i_nei, :])
#             if theta > np.pi/2.0:
#                 edges = np.concatenate((edges, np.array([i_node, i_nei, -np.log10(theta/np.pi)*phi]).reshape(1, 3)), axis=0)
#             else:
#                 beta = 1 + np.linalg.norm(np.dot(normals[i_node, 0:3], normals[i_nei, 0:3]))
#                 edges = np.concatenate((edges, np.array([i_node, i_nei, -beta*np.log10(theta/np.pi)*phi]).reshape(1, 3)), axis=0)
# edges = np.delete(edges, 0, 0)
# edges[:, 2] *= lambda_c*round_factor #scaling weights up
# edges = edges.astype(np.int32)

# #apply graph cuts algorithm
# refine_labels = pygco.cut_from_graph(edges, unaries, pairwise)
# refine_labels = refine_labels.reshape([-1, 1])


# #format labels
# labDf = pd.DataFrame({"toothNum": refine_labels.flatten()})
# #wires for labeling seem to be crossed somewhere, problem for later
# labDf["toothNum"] = labDf["toothNum"].astype(str)
# labDf["toothNum"] = labDf["toothNum"].replace("0", "gum")


# #join to color frame
# colFrame = cnf.colorNumFrame("U")
# colFrame = colFrame.drop(columns = ["fdiNum"])
# labDf = pd.merge(labDf, colFrame, on = "toothNum", how = "left")
# labDf["arch"] = "upper"
# labDf["alpha"] = 255

# #get vertex data from scan
# meshDat = raf.readAndFormat(segMeshPath, arch = "U")
# vertInfo = meshDat["face"][["vertex_indices"]]

# #join new label data to vertex data
# newFace = vertInfo.join(labDf, how = "left")

# #reformat
# newFace = newFace[["vertex_indices", "red", "green", "blue", "alpha", "color", "toothNum", "arch"]]