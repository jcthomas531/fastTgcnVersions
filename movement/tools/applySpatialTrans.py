import numpy as np
import pandas as pd

#apply a spatial transformation matrix to data formatted into the usual vert and face format
#if the transformation is for a single tooth, the dat should be for that tooth only
#see restrictMeshToTooth() in main tools directory

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