import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import trimeshToDf_labels as tdl
import trimesh
import pickle
import plyFunctions as pf
import centroidDistance as centDist



#pull variables from snakemake
preDPath = sys.argv[1] #the preD scan
finPath = sys.argv[2] #the fin scan
transPath = sys.argv[3] #the transformation
outFile = sys.argv[4] #where to output the distance and centroid data



#testing
# preDPath = "K:/iowaRme/preDelivAndFinalScans/preDelivScanU/dec016OriSeg/pat001u_preD_dec016Ori_seg.ply"
# finPath = "K:/iowaRme/preDelivAndFinalScans/finalScanU/dec016OriSeg/pat001u_fin_dec016Ori_seg.ply"
# transPath = "K:/iowaRme/registTrans/preDFin_dec016/pat001u_registTrans_dec016.pkl"
# outPath = "K:/iowaRme/movement/preDFin_dec016/test.csv"


#load in meshes
preDMesh = trimesh.load(preDPath)
finMesh = trimesh.load(finPath)

#load in transformation
openTransPath = open(transPath, "rb")
trans = pickle.load(openTransPath)
openTransPath.close()

#apply transformation
#the trimesh.apply_transform() actually occurs in place so we must copy 
#the object
finMeshTrans = finMesh.copy()
finMeshTrans.apply_transform(trans)


#convert preDMesh and finMeshTrans to dataframes
preDDf = tdl.trimeshToDf_labels(preDMesh)
finDf = tdl.trimeshToDf_labels(finMeshTrans)

#add in the tooth number variables for the faces
preDDf["face"] = pf.toothVars(preDDf["face"], arch = "U")
finDf["face"] = pf.toothVars(finDf["face"], arch = "U")

#calculate centroids
moveDist = centDist.centroidDistance(preDDf, finDf)

#export distances
moveDist.to_csv(outFile, index = False)



# import pyvista as pv
# import numpy as np
# def registAndCentPlot(scan1File, scan2File, cent):
#     #read in data
#     scan1 = pf.readAndFormat(file = scan1File, arch = "U")
#     scan2 = pf.readAndFormat(file = scan2File, arch = "U")
    
#     #get surface
#     surf1 = pf.giveSurf(face = scan1["face"], vertex = scan1["vert"])
#     surf2 = pf.giveSurf(face = scan2["face"], vertex = scan2["vert"])
    
#     #plot
#     plot1 = pv.Plotter()
#     plot1.add_mesh(surf1, scalars = "rgba", rgb = True)
#     plot1.add_mesh(surf2, color = "red", opacity = .5)
#     #plot1.add_mesh(surf2, scalars = "rgba", rgb = True, opacity = .5)
#     #add points for centroids at first time point
#     plot1.add_points(np.array(cent.loc[:,["x_pre", "y_pre", "z_pre"]]),
#                         color = "black", point_size=10,
#                         render_points_as_spheres=True)
#     #add points for centroids at second time point
#     plot1.add_points(np.array(cent.loc[:,["x_post", "y_post", "z_post"]]),
#                         color = "red", point_size=10,
#                         render_points_as_spheres=True)
#     plot1.show()
    
    
# registAndCentPlot(scan1File = preDPath,
#                   scan2File = finPath,
#                   cent = moveDist)