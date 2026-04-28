#function for decimating a mesh 


import pyvista as pv
import trimesh


def myDecimate(inFile, 
               outFile,
               nFace = 16000):
    
    #read in file
    meshOrig = pv.read(inFile)
    
    #number of faces in input mesh
    nFaceOrig = meshOrig.n_faces_strict
    
    #get reduction proportion using desired number of faces
    reduct = 1-(nFace/nFaceOrig)
    
    #decimate
    #there is decimate and there is decimate_pro
    #both have many options that you could get into
    #decimate uses VTK method
    #decimate_pro used another method
    #from my limited research it seems like VTK is pretty standard so I am going
    #to use the vinilla version of that. it would be interesting to compare it
    #with other methods or settings of decimation
    #heres a good tutorial on decimation with pyvista https://docs.pyvista.org/examples/01-filter/decimate.html
    
    #changing this to decimate_pro to match what I did with the teeth3ds data
    
    
    #perform decimation
    meshDec = meshOrig.decimate_pro(reduct)
    print(inFile, ": Mesh decimated from ", nFaceOrig, " to ",
          meshDec.n_faces_strict, " faces.",
          sep="")
    
    #output mesh as plyfile
    meshDec.save(outFile, binary = False)
    return meshDec


#testing
# import os
# os.chdir("K:/iowaRme")
# myDecimate("convertedPlyTestU.ply", 16000, "myDecimateFunctionTest.ply")
    



#decimate function specifically designed for use in the tooth3DS conversion process
#it is essentially just decimate pro with reduction calc and the function accepts
#a pyvista object instead of a file path so it could be used for other things too
#using decimate pro to retain point labels (if i am remembering correctly)
def decimate3DS(x, nFace = 16000):
    #number of faces in input mesh
    nFaceOrig = x.n_faces_strict
    
    #get reduction proportion using desired number of faces
    reduct = 1-(nFace/nFaceOrig)
    
    #perform decimation
    xDec = x.decimate_pro(reduct)
    
    return(xDec)



#this deimate function is designed to supercede the two above decimate functions
#it will read in a ply file, decimate it, and return a trimesh object that can then 
#either be exported or worked with further
#
#inFile, the ply file to be decimated
#nFace, the number of faces to decimate to
def decim(inFile, nFace = 16000):
    #read in file
    meshOrig = pv.read(inFile)
    
    #number of faces in input mesh
    nFaceOrig = meshOrig.n_faces_strict
    
    #get reduction proportion using desired number of faces
    reduct = 1-(nFace/nFaceOrig)
    
    #perform decimation
    meshDec = meshOrig.decimate_pro(reduct)
    
    #transform to a trimesh
    trimeshDec = pv.to_trimesh(meshDec)
    
    return trimeshDec

#example
# mesh1 = decim("K:/iowaRme/preDelivAndFinalScans/preDelivScanU/fullScans/pat001u_preD.ply")
# import sys
# sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
# import formatAndExport as fe
# m1Vert, m1Face = fe.trimeshToDfNoLabels(mesh1)
# fe.dfToPlyExport(vertDf = m1Vert,
#                  faceDf = m1Face,
#                  outFile = "K:/iowaRme/testDir/decimTest.ply")
















