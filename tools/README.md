# General

Directory for holding functions and scripts that are useful in the segmentation and measurement process.

# Files

## convertPlyFiles.py

Originally designed to take the ply files of RME scans downloaded from itero and convert them into the same format as the IOSSeg data so they can be used in the modeling process without changing any of the code. This is now morphed into a general purpose function for converting ply files into the IOSSeg format. Currently, it will accept ply files of the following formats and return ply files formatted like the IOSSeg files:

* scans of RME patients downloaded from itero
* ply files output from the registration process (uses open3d which has a slightly different output format)

NOTE: This function was designed to be used on test data. Thus, all of the color information that is stored in the faces will be overwritten with something uninformative. Also, RME scans from itero have color encoded in the vertices. This is removed as well. More work is needed in order to use this on files where the color information is important and must be retained


## registration.ply

Functions useful for determining the transformation that registers two scans. 

NOTE: This uses open3d which may or may not keep the color information in the faces when exporting the new mesh. Before using this function on training data or anything where retaining the color information in the faces is important, ensure that the output is what we want.

Example workflow:

```
#iowaRmeData
os.chdir("K:/iowaRme/fullScans/pat058")
#second scan will serve as source, what is being transformed
u058_12 = o3d.io.read_point_cloud("pat058u_12.ply")
#first scan will serve as target
u058_01 = o3d.io.read_point_cloud("pat058u_01.ply")
#plot how it looks prior to transformation (with actual colors)
o3d.visualization.draw_geometries([u058_12, u058_01])
monochromePlot(u058_12, u058_01)
#calculate registration
reg01_12 = getRegistration(source = u058_12, target=u058_01)
#look at stats on the registration performance
reg01_12
#look at transformation matri
reg01_12.transformation
#transform source to align with target
u058_12Trans = u058_12.transform(reg01_12.transformation)
#plot registered point clouds
monochromePlot(u058_12Trans, u058_01)

#applying the transformation to the file as a mesh and then exporting
#i will also load in the target as a mesh just to show that the alignment
#process works
u058_12Mesh = o3d.io.read_triangle_mesh("pat058u_12.ply")
u058_01Mesh = o3d.io.read_triangle_mesh("pat058u_01.ply")
o3d.visualization.draw_geometries([u058_12Mesh, u058_01Mesh])
monochromePlot(u058_12Mesh, u058_01Mesh)
#now we apply the transformation that was calculated for the point cloud
u058_12MeshTrans = u058_12Mesh.transform(reg01_12.transformation)
o3d.visualization.draw_geometries([u058_12MeshTrans, u058_01Mesh])
monochromePlot(u058_12MeshTrans, u058_01Mesh)
#then it can be exported
os.chdir("K:/testDir")
o3d.io.write_triangle_mesh("regiOutTest.ply", u058_12MeshTrans, write_ascii = True)
```