import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import registrationFuns as regi
import open3d as o3d


def plyToRegistTransformation(targetFile, sourceFile):
    #bring in the meshs as point clouds
    targetCloud = o3d.io.read_point_cloud(targetFile)
    sourceCloud = o3d.io.read_point_cloud(sourceFile)
    #obtrain the registration transformation
    reg = regi.getRegistration(source = sourceCloud, target=targetCloud)
    return reg