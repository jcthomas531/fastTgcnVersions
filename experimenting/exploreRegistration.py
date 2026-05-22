#exploring registration methods
import sys
import open3d as o3d
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import toothCentroids as tc
import plyFunctions as pf
import numpy as np
import copy
import registrationFuns as regi


#source is what will be moving
#target is what we are wanting to match to
sourceFile = "K:/iowaRme/preDelivAndFinalScans/finalScanU/dec016OriSeg/pat001u_fin_dec016Ori_seg.ply"
targetFile = "K:/iowaRme/preDelivAndFinalScans/preDelivScanU/dec016OriSeg/pat001u_preD_dec016Ori_seg.ply"
#files as o3d point clouds, target is t, source is s
tCloud = o3d.io.read_point_cloud(targetFile)
sCloud = o3d.io.read_point_cloud(sourceFile)
#coloring
tCloudColor = copy.deepcopy(tCloud)
tCloudColor.paint_uniform_color([0, 0.651, 0.929])
sCloudColor = copy.deepcopy(sCloud)
sCloudColor.paint_uniform_color([1, 0.706, 0])



# Create coordinate axes
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=10.0,   # axis length
    origin=[0, 0, 0]
)


###############################################################################
#lets start with a very basic approach, can we match on a smaller area defined
#on a region surrounding the gum centroid. we want to use the same registration
#tool that we have been using, dont reinvent the wheel to start.
#if we can identify the region around the gum centroid, we have a foothold into
#weighting based on proximity to the centroid.
#load in data as data frames
tDat = pf.readAndFormat(targetFile, arch = "U")
sDat = pf.readAndFormat(sourceFile, arch = "U")

#calculate centroids
tCent = tc.toothCentroids(face = tDat["face"], vertex = tDat["vert"]) 
sCent = tc.toothCentroids(face = sDat["face"], vertex = sDat["vert"]) 


#single point point clouds for each of the gum centroids
#this process is a bit like piping
#for target
tGumCent = (tCent.loc[tCent["toothNum"] == "gum", ["x", "y", "z"]]
            .iloc[0]
            .to_numpy())
tGumCentCloud = o3d.geometry.PointCloud()
tGumCentCloud.points = o3d.utility.Vector3dVector([tGumCent])
tGumCentCloud.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
#for source
sGumCent = (sCent.loc[sCent["toothNum"]=="gum", ["x", "y", "z"]]
            .iloc[0]
            .to_numpy()
            )
sGumCentCloud = o3d.geometry.PointCloud()
sGumCentCloud.points = o3d.utility.Vector3dVector([sGumCent])
sGumCentCloud.colors = o3d.utility.Vector3dVector([[0, 1, 0]])





#make into function
#takes vertDat as a data frame of the vertex data formatted as we have been 
#using it with pf.readAndFormat with ["vert"]
#takes centroidDat as a data frame of the tooth centroids formatted like the output
#from tc.toothCentroids
#returns a open3d point cloud object of the central region
def centralRegion3dRadius(vertDat, centroidDat, radius):
    
    #make the gum centroid into a numpy array
    #this process is a bit like piping
    gumCent = (centroidDat.loc[centroidDat["toothNum"] == "gum", ["x", "y", "z"]]
                .iloc[0]
                .to_numpy())
    
    #l2 of the difference between each point and the centroid
    vertNp = vertDat[["x", "y", "z"]].to_numpy()
    diffNorm = np.linalg.norm(vertNp - gumCent, axis = 1)
    
    #only keep in points that are within a certain number of units from the centroid
    #this is based on all 3 dimensions
    vertCentralRegion = vertDat.loc[diffNorm <= radius].copy()
    
    #make this dataframe into a open3d point cloud
    vertCRCloud =  o3d.geometry.PointCloud()
    vertCRCloud.points = o3d.utility.Vector3dVector(
        vertCentralRegion[["x", "y", "z"]].to_numpy()
        )
    
    return vertCRCloud


#testing
# tCR = centralRegion3dRadius(vertDat = tDat["vert"], centroidDat = tCent, radius = 10)
# tCR.paint_uniform_color([1,0,0])
# o3d.visualization.draw_geometries([tCR, tCloudColor,tGumCentCloud, axis])


#same things as above but for x and y dimensions
def centralRegion2dRadius(vertDat, centroidDat, radius):
    
    #make the gum centroid into a numpy array
    #this process is a bit like piping
    gumCent = (centroidDat.loc[centroidDat["toothNum"] == "gum", ["x", "y"]]
                .iloc[0]
                .to_numpy())
    
    #l2 of the difference between each point and the centroid
    vertNp = vertDat[["x", "y"]].to_numpy()
    diffNorm = np.linalg.norm(vertNp - gumCent, axis = 1)
    
    #only keep in points that are within a certain number of units from the centroid
    #based on only the x and y dimensions
    vertCentralRegion = vertDat.loc[diffNorm <= radius].copy()
    
    #make this dataframe into a open3d point cloud
    vertCRCloud =  o3d.geometry.PointCloud()
    vertCRCloud.points = o3d.utility.Vector3dVector(
        vertCentralRegion[["x", "y", "z"]].to_numpy()
        )
    
    return vertCRCloud

#testing
# tCR2 = centralRegion2dRadius(vertDat = tDat["vert"], centroidDat = tCent, radius = 10)
# tCR2.paint_uniform_color([1,0,0])
# o3d.visualization.draw_geometries([tCR2, tCloudColor, tGumCentCloud])



#I THINK THESE TRANFROMATIONS ARE HAPPENING IN PLACE, THEY NEED TO BE DONE ON COPIES

#now lets use these functions to do registration and take a look that the result

#baseline registration
baselineReg = regi.getRegistration(source = sCloud, target=tCloud)
regi.monochromePlot(sCloud, tCloud)
#perform transformation
sCloudTrans = sCloud.transform(baselineReg.transformation)
regi.monochromePlot(sCloudTrans, tCloud)

#3d radius
tCR3 = centralRegion3dRadius(vertDat = tDat["vert"], centroidDat = tCent, radius = 10)
sCR3 = centralRegion3dRadius(vertDat = sDat["vert"], centroidDat = sCent, radius = 10)
#visualization of central regions
tCR3.paint_uniform_color([1,0,0])
o3d.visualization.draw_geometries([tCR3, tCloudColor,  tGumCentCloud])
sCR3.paint_uniform_color([0,0,0])
o3d.visualization.draw_geometries([sCR3, sCloudColor,  sGumCentCloud])
#visualizing the central regions together
o3d.visualization.draw_geometries([tCR3, sCR3])
#obtain registration for central regions
reg3d = regi.getRegistration(sCR3, tCR3)
#apply transformation to source arch
sCloudTrans3d = sCloud.transform(reg3d.transformation) #ONE THING WE NEED TO MAKE SURE OF IS THAT THIS DOESNT HAPPEN IN PLACE LIKE IT DOES WITH TRIMESH
#visualize
regi.monochromePlot(sCloudTrans3d, tCloud)

#2d radius
tCR2 = centralRegion2dRadius(vertDat = tDat["vert"], centroidDat = tCent, radius = 10)
sCR2 = centralRegion2dRadius(vertDat = sDat["vert"], centroidDat = sCent, radius = 10)
#visualization of central regions
tCR2.paint_uniform_color([1,0,0])
o3d.visualization.draw_geometries([tCR2, tCloudColor, tGumCentCloud])
sCR2.paint_uniform_color([0,0,0])
o3d.visualization.draw_geometries([sCR2, sCloudColor,  sGumCentCloud])
#visualizing the central regions together
o3d.visualization.draw_geometries([tCR2, sCR2])
#obtain registration for central regions
reg2d = regi.getRegistration(sCR2, tCR2)
#apply transformation to source arch
sCloudTrans2d = sCloud.transform(reg2d.transformation) #ONE THING WE NEED TO MAKE SURE OF IS THAT THIS DOESNT HAPPEN IN PLACE LIKE IT DOES WITH TRIMESH
#visualize
regi.monochromePlot(sCloudTrans2d, tCloud)



###############################################################################



