import pyvista as pv
import copy
#decimate a pyvista object to a certain number of faces
#x is a pyvista object
#nFaces is the number of faces to decimate to
#was called decimate3DS in previously used decimationFuns.py, transitioning to 
#stand alone function scripts
def decim(x, nFace = 16000):
    xCopy = copy.deepcopy(x)
    #number of faces in input mesh
    nFaceOrig = xCopy.n_faces_strict
    
    #get reduction proportion using desired number of faces
    reduct = 1-(nFace/nFaceOrig)
    
    #perform decimation
    xDec = xCopy.decimate_pro(reduct)
    
    return(xDec)