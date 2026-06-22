import pandas as pd
import pyvista as pv
import numpy as np
#a function like plotArch that returns the surface instead of the image so that
#you can manipulate it and plot it later
def giveSurf(face, vertex):
    
    #copy the dataframes so it doesnt edit in place
    faceC = face.copy()
    vertexC = vertex.copy()
    
    #faces
    #get the number of vertexes for each shape (3 here bc triangles)
    faceC["nVert"] = faceC["vertex_indices"].apply(len)
    #make the vertices for the shape into columns
    faceCExpand = pd.DataFrame(faceC["vertex_indices"].tolist(),
                 columns=["v1", "v2", "v3"])
    faceC = faceC.join(faceCExpand)
    #order the data in the way that pyvista expects it and remove extra pieces
    #the color codes come at a different step
    faceCPV = faceC[["nVert", "v1", "v2", "v3"]]
    faceCPV = faceCPV.to_numpy()
    
    #vertices
    #extract the relavent columns
    #this could also be done with the normalized coordinates if you want
    vertexC = vertexC[["x", "y", "z"]]
    #make it how pyvista likes
    vertexC = vertexC.to_numpy()
    
    #use the vertex and face information to form the mesh
    surf = pv.PolyData(vertexC, faceCPV)
    
    #colors
    colors_ = faceC[["red", "green", "blue", "alpha"]]
    colors_ = colors_.to_numpy()
    #add the color information to the mesh
    surf.cell_data["rgba"] = colors_
    
    return surf

#example
# import os
# os.chdir("P:\\cph\\BIO\\Faculty\\gown\\research\\ThesisProjects\\Thomas\\IOSSegData\\train")
# l76 = plyRead("076_L.ply")
# l76["face"] = toothVars(l76["face"], arch = "L")
# s1 = giveSurf(face = l76["face"], vertex = l76["vert"])
# plotTest = pv.Plotter()
# plotTest.add_mesh(s1, scalars = "rgba", rgb = True)
# plotTest.show()