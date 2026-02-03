#rather than rewriting the get_data function, i think it will be easier to reformat the 
#rme style ply functions to look like those in the current data set and use the 
#existing framework via test_semseg
#this will probably involve some fudging of things like labels as these are not
#annotated samples but that can be done

###############################################################################
#functionized
###############################################################################


import sys
sys.path.append("/Users/jthomas48/dissModels/fastTgcnVersions/tools")
# import plyFunctions as pf
import numpy as np
from  plyfile import PlyData, PlyElement

def convertPly(inFile,
               outFile):
    
    #read in file to be converted
    inPly = PlyData.read(inFile)
    
    ####
    #Vertex conversion
    ####
    #obtain vertex information
    inVertex = inPly["vertex"]
    #create a list of only the items we want to keep
    toKeep = [name for name in inVertex.data.dtype.names
                  if name not in ('red', 'green', 'blue')]
    #build new vertex array, create an empty np array with the correct size and the
    #data types of the pieces that we are keeping
    newVert = np.empty(inVertex.count,
                       dtype=[(name, inVertex.data.dtype[name]) for name in toKeep])
    #fill the empty arrat with the data we are keeping
    for name in toKeep:
        newVert[name] = inVertex.data[name]
    #prepare as ply element
    newVertReady = PlyElement.describe(newVert, 'vertex')
    
    ####
    #Face conversion
    ####
    inFace = inPly["face"]
    #list of variables and their data types
    faceDataType = [
        ('vertex_indices', inFace.data.dtype['vertex_indices']),
        ('red',   'u1'),
        ('green', 'u1'),
        ('blue',  'u1'),
        ('alpha', 'u1'),
        ]
    #create empty array to store data in with correct size and type
    newFace = np.empty(inFace.count,
                       dtype=faceDataType)
    #input the vertex infromation
    newFace["vertex_indices"] = inFace.data["vertex_indices"]
    # fill RGBA with 255
    newFace['red']   = 255
    newFace['green'] = 255
    newFace['blue']  = 255
    newFace['alpha'] = 255
    #prepare as ply element
    newFaceReady = PlyElement.describe(newFace, 'face')
    
    
    ####
    #Output converted ply
    ####
    #i am going to have it output as ascii right now bc that what the other files are
    #like but i dont think it will matter in the read-ins. worth trying them in binary 
    #at some point bc it should be faster for larger files
    convPly = PlyData([newVertReady, newFaceReady], text = True)
    convPly.write(outFile)
    print(inFile, "\nconverted to desired ply format and written out to\n", outFile, sep = "")



#testing
#using raw ply from itero
# rmePath = "H:/schoolFiles/dissertation/rmeData/OrthoCAD_Export_261736634/261736634_shell_teethup_u.ply"
# convertPly(rmePath,
#            "H:/schoolFiles/dissertation/rmeData/OrthoCAD_Export_261736634/convFunTestU.ply")
# #using decimated files
# decPath = "K:/iowaRme/myDecimateFunctionTest.ply"
# convertPly(decPath,
#            "K:/iowaRme/convFunTestDec.ply")





