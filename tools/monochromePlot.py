import open3d as o3d
import copy

#largely stolen from these tutorials
#https://www.open3d.org/docs/0.7.0/tutorial/Basic/icp_registration.html
#https://www.open3d.org/docs/0.7.0/tutorial/Advanced/global_registration.html#global-registration
#see my notes in registrationLearning2.py for more details on the functions
#to apply a transformation use pointCloud.transform(transMat)
#open3d will not export face colors, must use trimesh


#see some previous work in tools/x_archive/registration


###############################################################################
#MONOCHROME PLOTTING
###############################################################################
#plot both point clouds, slight moditication of draw_registration_result from tutorial
#takes two point clouds, source and target
def monochromePlot(source, target):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([source_temp, target_temp])
    
    
#example
#works with both point clouds and meshes from o3d
# mesh1 = o3d.io.read_triangle_mesh("registerTest.ply")
# mesh2 = o3d.io.read_triangle_mesh(iossegPath)
# monochromePlot(mesh1, mesh2)