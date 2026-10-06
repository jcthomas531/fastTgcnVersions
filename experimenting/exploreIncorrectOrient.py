import pyvista as pv
import sys
sys.path.append("tools")
sys.path.append("tools/processes")
import getRotToMaster as grtm

#removing the scaling from the workflow has caused the orientation to master step to not work
#this is exploring why that may be and what can be done to fix it

mastPath = "K:/masterArches/masterArch1/mA1Full.ply"
t3dsPath = "K:/teeth3DS/scanData/upperPly_cSOriMastRemesh/01J54NZ0_U_cSOriMastRemesh.ply"
iosPath = "K:/IOSSegData/scanData/cleanU_cSOriMast/003_U_cSOriMast.ply"
iowaPath = "K:/iowaExpTest/scanData/rugAnnotForm_cSOriMastRemesh/pre/pat001Pre_formCSOriMastRemesh.ply"


mastMesh = pv.read(mastPath)
t3dsMesh = pv.read(t3dsPath)
iosMesh = pv.read(iosPath)
iowaMesh = pv.read(iowaPath)


def overlayPlot(mesh1, mesh2):
    op = pv.Plotter()
    op.add_mesh(mesh1, color = "bisque")
    op.add_mesh(mesh2, color = "lightsteelblue")
    op.show()

#t3ds is facing wrong way but seems to be the correct approximate size, if a little small
# overlayPlot(mastMesh, t3dsMesh)

#ios also seems to be approximently the corrects size but obviously incorrectly oriented
# overlayPlot(mastMesh, iosMesh)

#iowa is updside down and backwards but still approximently the right size
# overlayPlot(mastMesh, iowaMesh)



#all of the scans still seem to be of comprable size, so why does the orientation no longer work?

#trying rotation for each arch
t3dsRot = grtm.getRotToMaster(filePath=t3dsPath, masterArchPath=mastPath)
t3dsRot
#well this is obviously the reason, the rotation matrix is the identity matrix
#now we must find out why and correct
#confirming for all of the scan types
grtm.getRotToMaster(filePath=iosPath, masterArchPath=mastPath)
grtm.getRotToMaster(filePath=iowaPath, masterArchPath=mastPath)
#yes, this is the problem

#perhaps voxel size is now not appropriate for the size
import getRotToMaster2 as grtm2
t3dsRot_new = grtm2.getRotToMaster2(filePath=t3dsPath, masterArchPath=mastPath, voxel_size=5)
t3dsMesh_new = t3dsMesh.transform(t3dsRot_new, inplace = False)
overlayPlot(mesh1 = mastMesh, mesh2 = t3dsMesh_new)
#this larger voxel size appears to be working


def voxelAndPlot(fp, v = 5, it = 100):
    mesh = pv.read(fp)
    rot = grtm2.getRotToMaster2(filePath=fp, masterArchPath=mastPath, voxel_size=v, iters=it)
    meshTrans = mesh.transform(rot, inplace = False)
    overlayPlot(mesh1 = mastMesh, mesh2 = meshTrans)


#trying for other scans
voxelAndPlot(fp = t3dsPath, v = 7, it = 400)
voxelAndPlot(fp = iosPath, v = 6, it = 400)
voxelAndPlot(fp = iowaPath, v = 6, it = 400)

#voxel size values
#v=1 works sometimes but is catastrophically bad sometimes
#v=2 seems more consistent but still not always great 
#v=3 still lacks some consistency
#v=4 always gives the correct orientation if not exactly the same each time
#v=5 seems to work exactly the same each time
#v=6 seems even better
#v=7 does not show meaningful improvement

#making sure this works with any t3ds scan
from pathlib import Path
import random

t3dsDir = Path("K:/teeth3DS/scanData/upperPly_cSOriMastRemesh/")
t3dsFile = random.choice(list(t3dsDir.iterdir()))
voxelAndPlot(fp = t3dsFile, v = 6, it = 400)
#t3ds seems to work

iosDir = Path("K:/IOSSegData/scanData/cleanU_cSOriMast/") 
iosFile = random.choice(list(iosDir.iterdir()))
voxelAndPlot(fp = iosFile, v = 6, it = 400)
#ios seems to work

iowaDir = Path("K:/iowaExpTest/scanData/rugAnnotForm_cSOriMastRemesh/post/")
iowaFile = random.choice(list(iowaDir.iterdir()))
voxelAndPlot(fp = iowaFile, v = 6, it = 400)
#seems to work on ours as well