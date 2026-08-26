from pyglet.libs.win32.constants import PROCESSOR_INTEL_386
import pickle
import sys
import pandas as pd
import pyvista as pv
import numpy as np

sys.path.append("tools")
import readAndFormat as raf
import toothCentroids as toCe
import giveSurf

prePath = "K:/iowaExpTest/segResults/segResults_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/pre/pat001Pre_formCSOriMastRemesh_seg.ply"
postPath = "K:/iowaExpTest/segResults/segResults_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/post/pat001Post_formCSOriMastRemesh_rugAnnotSuperimp_seg.ply"
transPath = "K:/iowaExpTest/spatialTrans/pat001SpatialTransMats.pkl"

#load in trans
with open(transPath, "rb") as f:
    trans = pickle.load(f)


#load formatted meshes
preDf = raf.readAndFormat(file = prePath, arch = "U")
postDf = raf.readAndFormat(file = postPath, arch = "U")

#find centroids
preCent = toCe.toothCentroids(face = preDf["face"], vertex = preDf["vert"])
postCent = toCe.toothCentroids(face = postDf["face"], vertex = postDf["vert"])

#surfaces for pyvista
preSurf = giveSurf.giveSurf(face = preDf["face"], vertex = preDf["vert"])
postSurf = giveSurf.giveSurf(face = postDf["face"], vertex = postDf["vert"])






import restrictMeshToTooth as rmtt
import importlib
importlib.reload(rmtt)
importlib.reload(giveSurf)
pre3 = rmtt.restrictMeshToTooth(preDf, toothNum = "3")
post3 = rmtt.restrictMeshToTooth(postDf, toothNum = "3")

pre3Surf = giveSurf.giveSurf(face = pre3["face"], vertex = pre3["vert"])
post3Surf = giveSurf.giveSurf(face = post3["face"], vertex = post3["vert"])

# p3 = pv.Plotter()
# p3.add_mesh(pre3Surf, color = "white")
# p3.add_mesh(post3Surf, scalars = "rgba", rgb = True,  opacity = .6)
# p3.show()



#function to apply spatial trans matrix to a tooth
#dat only including tooth of interest but in same format as readAndFormat()
#trans a numpy array
def applySpatialTrans(dat, trans):
    #set up vertex matrix
    vMat = np.transpose(dat["vert"][["x", "y", "z"]].to_numpy())
    bottomRow = [1] * np.shape(vMat)[1]
    vMatReady = np.vstack([vMat, bottomRow])
    #multiply and format result
    newVertMat = np.dot(trans, vMatReady)
    newVertMatForm = np.transpose(np.delete(newVertMat, (3), axis = 0))
    #put new vertex information with face infromation
    #could add normals but probably dont need to right now
    rigid = {}
    rigid["vert"] = pd.DataFrame(newVertMatForm, columns=["x", "y", "z"])
    rigid["face"] = dat["face"]
    return rigid
#testing
# import readAndFormat as raf
# import restrictMeshToTooth as rmtt
# prePath = "K:/iowaExpTest/segResults/segResults_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/pre/pat001Pre_formCSOriMastRemesh_seg.ply"
# preDf = raf.readAndFormat(file = prePath, arch = "U")
# pre3 = rmtt.restrictMeshToTooth(preDf, toothNum = "3")
# #
# transPath = "K:/iowaExpTest/spatialTrans/pat001SpatialTransMats.pkl"
# with open(transPath, "rb") as f:
#     allTrans = pickle.load(f)
# t3Trans = allTrans["3"]
# #
# test = applySpatialTrans(dat = pre3, trans=t3Trans)


#now apply the transformation and visualize


#read in transformation
transPath = "K:/iowaExpTest/spatialTrans/pat001SpatialTransMats.pkl"
with open(transPath, "rb") as f:
    allTrans = pickle.load(f)
t3Trans = allTrans["3"]


# #set up vertex matrix
# aaa = np.transpose(pre3["vert"][["x", "y", "z"]].to_numpy())
# bottomRow = [1] * np.shape(aaa)[1]
# aaaReady = np.vstack([aaa, bottomRow])

# #multiply and format result
# new3 = np.dot(t3Trans, aaaReady)
# new3Form = np.transpose(np.delete(new3, (3), axis = 0))

# #put new vertex information with face infromation
# #could add normals but probably dont need to right now
# rigid3 = {}
# rigid3["vert"] = pd.DataFrame(new3Form, columns=["x", "y", "z"])
# rigid3["face"] = pre3["face"]

rigid3 = applySpatialTrans(dat = pre3, trans=t3Trans)

r3Surf = giveSurf.giveSurf(face = rigid3["face"], vertex = rigid3["vert"])

pR3 = pv.Plotter()
# pR3.add_mesh(pre3Surf, color = "white")
pR3.add_mesh(post3Surf, color = "white",  opacity = 1)
pR3.add_mesh(r3Surf, color = "blue",  opacity = .6)
pR3.show()


#it looks like what we have here is the transfromation FROM time 2 TO time 1 applied to time 1
#potentnailly we are working the wrong direction in our ICP

len(allTrans)
range(len(allTrans))

movedDict = {}
patTeeth = list(allTrans.keys())
for i in range(len(patTeeth)):
    #tooth number for this iteration
    ti = patTeeth[i]
    #data for ith tooth
    datTi = rmtt.restrictMeshToTooth(preDf, toothNum = ti)
    #get transformation matrix for ith tooth
    transTi = allTrans[ti]
    #perform transformation and export
    movedDict[ti] = applySpatialTrans(dat = datTi, trans=transTi)

surfDict = {
    tooth: giveSurf.giveSurf(face = dat["face"], vertex = dat["vert"]) 
    for tooth, dat in movedDict.items()
    }
surfDictC = surfDict.copy()
del surfDictC["gum"]



#plot
#time 1 and time 2
pPrePost = pv.Plotter()
pPrePost.add_mesh(preSurf, color = "white",  opacity = 1)
pPrePost.add_mesh(postSurf, color = "green",  opacity = .6)
pPrePost.show()

#rigid registration and t1
pMove = pv.Plotter()
pMove.add_mesh(preSurf, color = "white",  opacity = 1)
for i in range(len(patTeeth)-1):
    pMove.add_mesh(list(surfDictC.values())[i], color = "blue",  opacity = .6)
pMove.show()

#rigid registration and true t2
pTruth = pv.Plotter()
pTruth.add_mesh(postSurf, color = "grey",  opacity = 1)
for i in range(len(patTeeth)-1):
    pTruth.add_mesh(list(surfDictC.values())[i], color = "blue",  opacity = .6)
pTruth.show()

#rigid registartion, t1, true t2
pAll = pv.Plotter()
pAll.add_mesh(preSurf, color = "white",  opacity = 1)
for i in range(len(patTeeth)-1):
    pAll.add_mesh(list(surfDictC.values())[i], color = "blue",  opacity = .4)
pAll.add_mesh(postSurf, color = "green",  opacity = .6)
pAll.show()

