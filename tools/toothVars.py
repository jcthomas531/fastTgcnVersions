import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import colorNumFrame as cnf
import pandas as pd

#function that adds tooth number and other tooth characteristics to the face data frame
#face takes a face data frame that has been set up using plyRead
#arch takes a string "L" or "U" denoting the upper or lower arch
def toothVars(face, arch):
    #make copies of the dataframes so you dont edit in place
    faceC = face.copy()
    
    #identify if we have an upper or lower arch
    #get color and tooth number associations
    #merge with face
    #create variable in face distinguishing arch
    if arch == "U":
        
        uNumCol = cnf.colorNumFrame("U")
        uNumCol = uNumCol[["toothNum", "color"]] #subset for historical compatibility reasons
        
        faceC = faceC.merge(uNumCol, on="color", how = "left", validate = "many_to_one")
        faceC["arch"] = "upper"
    elif arch == "L":
        
        lNumCol = cnf.colorNumFrame("L")
        lNumCol = lNumCol[["toothNum", "color"]] #subset for historical compatibility reasons
        
        faceC = faceC.merge(lNumCol, on="color", how = "left", validate = "many_to_one")
        faceC["arch"] = "lower"
    else:
        raise ValueError("arch arguement must be either 'L' or 'U'")
    #return updated dataframe
    return faceC


#example
# import os
# os.chdir("P:\\cph\\BIO\\Faculty\\gown\\research\\ThesisProjects\\Thomas\\IOSSegData\\train")
# l07 = plyRead("007_L.ply")
# l07["face"] = toothVars(l07["face"], arch = "L")