import sys
sys.path.append("Y:/dissModels/fastTgcnVersions/tools/registration")
import registrationFuns as regi
import os
import open3d as o3d
import copy


#bring in an arbitrary iosseg scan to use as a target for registration
iossegDir = "K:/IOSSegData/clean/trainCleanU/"
regTarget = o3d.io.read_point_cloud(iossegDir + "007_U.ply")


#to start lets grab an arbitrary scan from the warmstartTest data
#this will be the source, what is being transformed
testDir = "K:/testDir/warmstartTestDataReg/"
os.chdir(testDir)
regSource = o3d.io.read_point_cloud("train/00OMSZGW_UDecim016.ply")


#
#plot how it looks prior to transformation (with actual colors)
o3d.visualization.draw_geometries([regSource, regTarget])
regi.monochromePlot(regSource, regTarget)
#calculate registration
reg1 = regi.getRegistration(source = regSource, target=regTarget)
#look at stats on the registration performance
reg1
#look at transformation matri
reg1.transformation
#transform source to align with target
regSourceTrans = regSource.transform(reg1.transformation)
#plot registered point clouds
regi.monochromePlot(regSourceTrans, regTarget)


#though not perfect bc these are two different scans, they are pointing in the same
#direction. One question about this is does this transformation allow for shrinking?
#is that something we would want or not want? I would think not want unless on different scales..
#also there are probably features we can tweak to make the fit a bit more general since
#these are two different arches but this is good for now.
#now we must export it and make sure its format is right ugh i really need to streamline the process