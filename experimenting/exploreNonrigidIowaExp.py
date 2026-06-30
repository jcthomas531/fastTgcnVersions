from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycpd import DeformableRegistration
import numpy as np



import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
from  plyfile import PlyData
import trimesh


#using full scans to start since this is what was superimposed
#this is a very large scan, lets see how it works, may need to use different version
prePath = "K:/iowaExpansion/fullRugaeAnnotScans/pre/pat001Pre_annot.ply"
postSiPath = "K:/iowaExpansion/superimposition/transPostScan/annotRugaeTransPostScan/pat001Post_annotRugaeSuperimp.ply"

#read in meshes
preMesh = trimesh.load_mesh(prePath, process = False)
postSiMesh = trimesh.load_mesh(postSiPath, process = False)

#extract vertex information
preVert = np.asarray(preMesh.vertices)
postSiVert = np.asarray(postSiMesh.vertices)


preMesh.show()

#down sample to 1000
import random
random.seed(826)
preSamp = random.sample(population=range(len(preVert)), k = 7000)
preVert = np.asarray(preVert[preSamp,])
postSiSamp = random.sample(population=range(len(postSiVert)), k = 7000)
postSiVert = np.asarray(postSiVert[postSiSamp,])



#registration
from pycpd import DeformableRegistration
regDeform =  DeformableRegistration(**{'X': preVert, 'Y': postSiVert})
aaa = regDeform.register(postSiVert)[0]



import pyvista as pv
p1 = pv.Plotter()
p1.add_points(preVert, color = "black")
#p1.add_points(postSiVert, color = "blue")
p1.add_points(aaa, color = "green")
p1.show()


dir(regDeform)
p1 = regDeform.get_registration_parameters()[0]
p2 = regDeform.get_registration_parameters()[1]
p1.shape
p2.shape
trans =  p1 @ p2
preVert

#need points and vectors
pdata = pv.vector_poly_data(preVert, trans)
pdata.point_data.keys()
pdata.glyph(orient='vectors', scale='mag').plot()






#testing
p = regDeform.P
p
import pandas as pd
pDf = pd.DataFrame(p)
pDf.max(axis=0)
pDf.max(axis=1)
max(pDf.max(axis=0))
max(pDf.max(axis=1))
#it seems we can extract the P matrix but it seems to be a matrix of 0s