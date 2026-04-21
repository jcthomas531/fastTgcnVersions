#allow imports from other directory
import sys
sys.path.append("/Users/jthomas48/dissModels/intraoralSegmentation/fastTgcnEasy/")



#path to where test ply files stored (SHOULD THIS BE WHERE THE NEW FILES ARE OR WHERE THE ORIGINAL TESTS FILES WHERE)
path1 = "/Shared/gb_lss/Thomas/iowaRme/testDir/test1"
#path1 = "/Shared/gb_lss/Thomas/trainTestSets/teeth3dsDecim016/test - Copy"




from dataloader import plydataset
set1 = plydataset(path = path1, arch = "u", mode = 'test', model = 'meshsegnet')
from torch.utils.data import DataLoader
loader1 = DataLoader(set1, batch_size=1, shuffle=True, num_workers=8)


#https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
from Baseline import Baseline
model1 = Baseline(in_channels=12, output_channels=17)
import torch
modelPath = "/Users/jthomas48/dissModels/intraoralSegmentation/fastTgcnEasy/modelOutputs/2026_04_21 teeth3dsFullTrainDecim016/checkpointsAndLogs/checkpoints/coordinate_140_0.939648.pth"
# model1.load_state_dict(torch.load(modelPath, weights_only=True))
# model1.cuda() #move model to gpu
# model1.eval()
#-------------------patch
state = torch.load(modelPath, map_location="cuda", weights_only=True)
model1.load_state_dict(state)
model1 = model1.cuda()
model1.train()  # <- IMPORTANT: use per-batch BN stats like training-time test_semseg
print("model.training =", model1.training)  # sanity check: should print True
#-------------------


#try test_semseg out of the box
# from utils import test_semseg
# import os
# os.chdir("/Shared/gb_lss/Thomas/iowaRme/test1Pred/")
# print(test_semseg(model = model1, loader = loader1, arch = "u", plyPath="/Shared/gb_lss/Thomas/iowaRme/test1Pred/",
#                                                       num_classes=17, generate_ply=True))
#this seems to be having a problem but mostly for some reason with the output path



#lets disect test_semseg and see where it gets caught

#load in packages
import os
import shutil
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

from dataloader import generate_plyfile, plydataset
from loss import IoULoss, DiceLoss


#arguements
model = model1
loader = loader1
arch = "u"
plyPath="/Shared/gb_lss/Thomas/iowaRme/testDir/test1PredD"
num_classes=17
generate_ply=True





#internals


iou_tabel = np.zeros((num_classes,3))
metrics = defaultdict(lambda:list())
dice_loss = DiceLoss()
hist_acc = []
macc = 0
mdice = 0



#getting the ply path ready for each use, this could probably be streamlined
plyPathStr = str(plyPath)
plyPathStr1 = "./" + plyPathStr.replace("\\", "/")
plyPathStr2 = plyPathStr.replace("\\", "/") + "/%s"

print(plyPathStr)
print(plyPathStr1)
print(plyPathStr2)


print("a")


from utils import compute_cat_iou, compute_mACC
#removed for now, this seems to be the first catch
# shutil.rmtree(plyPathStr1)
# os.mkdir(plyPathStr1)
print("b")
#
#i believe a lot of the stuff in here is unnecessary, we really just need the preds
#
for batch_id, (index, points, label_face, label_face_onehot, name, raw_points_face, idx_face) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
    batchsize, num_point, _ = points.size()
    point_face = raw_points_face[0].numpy()
    index_face = index[0].numpy()
    coordinate = points.transpose(2,1)
    normal = points[:, :, 12:]
    centre = points[:, :, 9:12]
    label_face = label_face[:, :, 0]
    print("c")
    coordinate, label_face, centre, idx_face = Variable(coordinate.float()), Variable(label_face.long()), Variable(centre.float()), Variable(idx_face.float())
    coordinate, label_face, centre, idx_face = coordinate.cuda(), label_face.cuda(), centre.cuda(), idx_face.cuda()
    print("d")
    with torch.no_grad():
        # pred, _ = model(coordinate, idx_face)
        pred = model(coordinate, idx_face)
        # pred = model(coordinate)
    print("dd")
    mdice += dice_loss(pred.max(dim=-1)[0], label_face)
    iou_tabel, iou_list = compute_cat_iou(pred,label_face,iou_tabel) 
    pred = pred.contiguous().view(-1, num_classes)
    label_face = label_face.view(-1, 1)[:, 0]
    pred_choice = pred.data.max(1)[1]
    macc += compute_mACC(pred_choice, label_face).cpu().data.numpy() 
    correct = pred_choice.eq(label_face.data).cpu().sum()
    metrics['accuracy'].append(correct.item()/ (batchsize * num_point))
    label_face = pred_choice.cpu().reshape(pred_choice.shape[0], 1)
    print("e")
    if generate_ply:

        #label_face=label_optimization(index_face, label_face)

        generate_plyfile(index_face = index_face, point_face = point_face,
                         label_face = label_face, arch = arch, path=(plyPathStr2) % name)
print("f")
iou_tabel[:,2] = iou_tabel[:,0] /iou_tabel[:,1]
# iou = np.where(iou_tabel<=1.)
hist_acc += metrics['accuracy']
metrics['accuracy'] = np.mean(metrics['accuracy'])
metrics['iou'] = np.mean(iou_tabel[:, 2])
iou_tabel = pd.DataFrame(iou_tabel,columns=['iou','count','mean_iou'])
iou_tabel['Category_IOU'] = ["label%d"%(i) for i in range(num_classes)]
cat_iou = iou_tabel.groupby('Category_IOU')['mean_iou'].mean()
mIoU = np.mean(cat_iou)

print("------------------finish--------------------------")



