#rather than rewriting the get_data function, i think it will be easier to reformat the 
#rme style ply functions to look like those in the current data set and use the 
#existing framework via test_semseg
#this will probably involve some fudging of things like labels as these are not
#annotated samples but that can be done




#the goal is to do something like this (from copilot), following the same general
#structure as is used in train.py

# import torch
# from torch.utils.data import DataLoader
# from Baseline import Baseline
# from dataloader import plydataset
# from utils import test_semseg

# device = "cuda" if torch.cuda.is_available() else "cpu"

# # 1) Load model
# model = Baseline(in_channels=12, output_channels=17).to(device)
# state = torch.load("/path/to/checkpoints/best.pth", map_location=device)
# model.load_state_dict(state)
# model.eval()

# # 2) Build dataset/loader pointing to folder of new PLYs
# arch = "l"  # or "u"
# data_dir = "./inference_samples"
# dataset = plydataset(path=data_dir, arch=arch, mode="test", model="meshsegnet")
# loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

# # 3) Run with test_semseg (will export PLYs to ./pred_global if generate_ply=True)
# with torch.no_grad():
#     metrics, mIoU, cat_iou, mAcc, _ = test_semseg(
#         model=model,
#         loader=loader,
#         arch=arch,
#         num_classes=17,
#         generate_ply=True
#     )



###############################################################################
import os
os.chdir("Y:/dissModels/fastTgcnVersions/tools/")
import plyFunctions as pf
import numpy as np
iosPath = "K:/IOSSegData/clean/testClean/001_L.ply"
rmePath = "H:/schoolFiles/dissertation/rmeData/OrthoCAD_Export_261736634/261736634_shell_teethup_u.ply"

from  plyfile import PlyData, PlyElement

iosPly = PlyData.read(iosPath)
rmePly = PlyData.read(rmePath)


#examine structure
print(iosPly)
print(rmePly)
#rme file has RGB values in vertex that need to be removed
#rme file has "property int object" in the first like of face that needs to be removed
#rme file has "property list uchar float texcoord" in last line of face that needs to be removed
#rme is missing RGB and alpha values in face that need to be fudged (try all 255)

#we can strip these with the plyfile package
#obtain vertex information
rmeVertex = rmePly["vertex"]
#create a list of only the items we want to keep
toKeep = [name for name in rmeVertex.data.dtype.names
              if name not in ('red', 'green', 'blue')]
#build new vertex array, create an empty np array with the correct size and the
#data types of the pieces that we are keeping
newVert = np.empty(rmeVertex.count,
                   dtype=[(name, rmeVertex.data.dtype[name]) for name in toKeep])
#fill the empty arrat with the data we are keeping
for name in toKeep:
    newVert[name] = rmeVertex.data[name]
   #prepare as ply element
newVertReady = PlyElement.describe(newVert, 'vertex')


#lets try something similar for the face infromation
rmeFace = rmePly["face"]
#list of variables and their data types
faceDataType = [
    ('vertex_indices', rmeFace.data.dtype['vertex_indices']),
    ('red',   'u1'),
    ('green', 'u1'),
    ('blue',  'u1'),
    ('alpha', 'u1'),
    ]
#create empty array to store data in with correct size and type
newFace = np.empty(rmeFace.count,
                   dtype=faceDataType)
#input the vertex infromation
newFace["vertex_indices"] = rmeFace.data["vertex_indices"]
# fill RGBA with 255
newFace['red']   = 255
newFace['green'] = 255
newFace['blue']  = 255
newFace['alpha'] = 255
#prepare as ply element
newFaceReady = PlyElement.describe(newFace, 'face')



#i am going to have it output as ascii right now bc that what the other files are
#like but i dont think it will matter in the read-ins. worth trying them in binary 
#at some point bc it should be faster for larger files
os.chdir("H:/schoolFiles/dissertation/rmeData/OrthoCAD_Export_261736634")
convPly = PlyData([newVertReady, newFaceReady], text = True)
convPly.write("convertedPlyTestU.ply")









