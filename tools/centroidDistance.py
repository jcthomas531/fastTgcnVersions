import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import toothCentroids as tc


###############################################################################
#GET CENTROIDS FOR TWO MESHES
###############################################################################

#mesh1 and mesh2 are objects formatted like plyRead() in plyFunctions.py
#ideall these are registered meshes
def  centroidDistance(mesh1, mesh2):
    
    
    #calculate each scans centroids
    mesh1Cent = tc.toothCentroids(face = mesh1["face"], vertex = mesh1["vert"])
    mesh2Cent = tc.toothCentroids(face = mesh2["face"], vertex = mesh2["vert"])
    
    #give the two time points suffixes for their centroid location columns
    toModify = mesh1Cent.columns[range(1,4)] #this will be the same for both data frames
    mesh1Cent = mesh1Cent.rename(columns={i: f"{i}_pre" for i in toModify})
    mesh2Cent = mesh2Cent.rename(columns={i: f"{i}_post" for i in toModify})
    
    #it is possible based on incorrect segmentation that one scan will have a different
    #number of segmented classes than another, i believe the arithmatic below will
    #be able to handle missing values
    
    #since it is possible to have a different number of classes in the two data 
    #frames, we must set the join up so that all of the classes are represented
    if (len(mesh1Cent) >=  len(mesh1Cent)):
        joinDirection = "left"
    elif (len(mesh1Cent) <  len(mesh1Cent)):
        joinDirection = "right"
    else:
        raise ValueError("Error in determining data frame lenghts prior to join")
    
    #join the two time points together
    #pat055uBothCent = pat055u_01Cent.merge(pat055u_16Cent, on = "toothNum", how = "left")
    scanBothCent = mesh1Cent.merge(mesh2Cent, on = "toothNum", how = joinDirection)
    
    #calculate the values for the distance vector between the two time points
    scanBothCent["xDiff"] = scanBothCent["x_pre"] - scanBothCent["x_post"]
    scanBothCent["yDiff"] = scanBothCent["y_pre"] - scanBothCent["y_post"]
    scanBothCent["zDiff"] = scanBothCent["z_pre"] - scanBothCent["z_post"]
    
    #find the l2 norm of the distance vector
    scanBothCent["l2Norm"] = (scanBothCent["xDiff"]**2 + scanBothCent["yDiff"]**2 + scanBothCent["zDiff"]**2) ** (1/2)
    
    #unit vector values for the distance vector
    scanBothCent["xDiffUnit"] = scanBothCent["xDiff"]/scanBothCent["l2Norm"]
    scanBothCent["yDiffUnit"] = scanBothCent["yDiff"]/scanBothCent["l2Norm"]
    scanBothCent["zDiffUnit"] = scanBothCent["zDiff"]/scanBothCent["l2Norm"]
    
    return scanBothCent
