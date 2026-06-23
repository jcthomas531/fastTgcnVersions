import numpy as np
import open3d as o3d
import copy

# mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
# T = np.eye(4)
# T
# T[:3, :3] = mesh.get_rotation_matrix_from_xyz((0, np.pi / 3, np.pi / 2))
# T
# T[0, 3] = 1
# T
# T[1, 3] = 1.3
# print(T)
# mesh_t = copy.deepcopy(mesh).transform(T)
# o3d.visualization.draw_geometries([mesh, mesh_t])





# def qr_full(num_samples=1):
#     z = np.random.randn(num_samples, 3, 3)
#     q, r = np.linalg.qr(z)
#     sign = 2 * (np.diagonal(r, axis1=-2, axis2=-1) >= 0) - 1
#     rot = q
#     rot *= sign[..., None, :]
#     rot[:, 0, :] *= np.linalg.det(rot)[..., None]
#     return rot

# aaa = qr_full()
# aaa

# tTest = np.eye(4)
# tTest[:3,:3] = aaa
# tTest

# #this is rotating around the origin which is not where the centroid of the scan
# #actually is. we should try centering and scaling each of the scans for consistency





import trimesh
prePath = "K:/iowaExpansion/segResults/segResults_remeshT3dsEpoch270/pre/pat001Pre_modelReady_seg.ply"
aaa = trimesh.load(prePath)

bbb = copy.deepcopy(aaa)
bbb.apply_translation(-bbb.centroid)
scaleFac = 1/np.max(bbb.extents)
bbb.apply_scale(scaleFac)
bbb.centroid
bbb.extents

ccc = copy.deepcopy(bbb)
ccc.apply_transform(trimesh.transformations.random_rotation_matrix())



aaa_o3d = o3d.geometry.TriangleMesh()
aaa_o3d.vertices = o3d.utility.Vector3dVector(aaa.vertices)
aaa_o3d.triangles = o3d.utility.Vector3iVector(aaa.faces)
aaa_o3d.compute_vertex_normals()

bbb_o3d = o3d.geometry.TriangleMesh()
bbb_o3d.vertices = o3d.utility.Vector3dVector(bbb.vertices)
bbb_o3d.triangles = o3d.utility.Vector3iVector(bbb.faces)
bbb_o3d.compute_vertex_normals()

ccc_o3d = o3d.geometry.TriangleMesh()
ccc_o3d.vertices = o3d.utility.Vector3dVector(ccc.vertices)
ccc_o3d.triangles = o3d.utility.Vector3iVector(ccc.faces)
ccc_o3d.compute_vertex_normals()



# o3d.visualization.draw_geometries([aaa_o3d, bbb_o3d, ccc_o3d])
o3d.visualization.draw_geometries([bbb_o3d, ccc_o3d])

#get this functionized and cotrolled with a seed
import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import trimeshExtractFaceLabels as tefl



#seed setting
# import os
# os.environ["OMP_NUM_THREADS"] = "1"

# random.seed(seed)
# np.random.seed(seed)
# o3d.utility.random.seed(seed)


seed = 826
np.random.seed(seed)
aaa = trimesh.transformations.random_rotation_matrix(num = 2)

writePath = "K:/teeth3DS/randomRotations/"
import pickle
for i in range(len(aaa)):
    filePath = open(writePath + str(i) + ".pkl", "wb")
    pickle.dump(obj = aaa[i],
                file = filePath)
    filePath.close()




#can be read in like: 
# with open("K:/teeth3DS/randomRotations/0.pkl", "rb") as f:
#     t0 = pickle.load(f)