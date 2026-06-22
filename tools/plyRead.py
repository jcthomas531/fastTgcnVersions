from  plyfile import PlyData
import pandas as pd

#function that reads in a ply file and formats in how I like
#this will require PlyData and pandas package
def plyRead(file):
    #pdb.set_trace()
    #read in the object
    plyObject = PlyData.read(file)
    #get the vertex and face data
    plyVert = pd.DataFrame(plyObject["vertex"].data)
    plyFace = pd.DataFrame(plyObject["face"].data)
    #create new variable in face data for tooth color that concats RGB vals
    plyFace["color"] = (plyFace["red"].astype(str).str.zfill(3) + "-" +
                         plyFace["green"].astype(str).str.zfill(3) + "-" +
                         plyFace["blue"].astype(str).str.zfill(3))
    #return a dictionary of the vertex and face information
    #this dictionary object seems like a named list in R
    return {"vert": plyVert, "face": plyFace}
#example
# import os
# os.chdir("P:\\cph\\BIO\\Faculty\\gown\\research\\ThesisProjects\\Thomas\\IOSSegData\\train")
# l76 = plyRead("076_L.ply")