#function for decimating a mesh 


import pyvista as pv

def myDecimate(inFile, 
               nFace,
               outFile):
    
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
    
    #perform decimation
    meshDec = meshOrig.decimate(reduct)
    print(inFile, ": Mesh decimated from ", nFaceOrig, " to ",
          meshDec.n_faces_strict, " faces.",
          sep="")
    
    #output mesh as plyfile
    meshDec.save(outFile, binary = False)


#testing
# import os
# os.chdir("K:/iowaRme")
# myDecimate("convertedPlyTestU.ply", 16000, "myDecimateFunctionTest.ply")
    
    


