import numpy as np
import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import plotArch
#function that highlights a series of tooth numbers highlights them in color
#this does not check to make sure the number requested is in that arch, but
#that would be pretty easy to add
#it would also be nice to have a version of this that also returned just the surface
def toothHighlightPlot(face, vertex, toothNums):
    #make copies of the dataframes so you dont edit in place
    faceC = face.copy()
    vertexC = vertex.copy()
    #make all colors besides the one we are looking at white
    faceC["red"] = np.where(faceC["toothNum"].isin(toothNums), faceC["red"], 255)
    faceC["green"] = np.where(faceC["toothNum"].isin(toothNums), faceC["green"], 255)
    faceC["blue"] = np.where(faceC["toothNum"].isin(toothNums), faceC["blue"], 255)
    return plotArch.plotArch(face = faceC, vertex = vertexC)
#example
# import os
# os.chdir("P:\\cph\\BIO\\Faculty\\gown\\research\\ThesisProjects\\Thomas\\IOSSegData\\train")
# l76 = plyRead("076_L.ply")
# l76["face"] = toothVars(l76["face"], arch = "L")
# toothHighlightPlot(l76["face"], l76["vert"], ["17", "30"])