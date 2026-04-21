


#some tutorials
#https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html



#when i import this function, i am not sure that it will run these packages up 
#at the top, i might need to look into a way to do that
import sys
sys.path.append("/Users/jthomas48/dissModels/intraoralSegmentation/fastTgcnEasy/")
from dataloader import plydataset
from torch.utils.data import DataLoader
from Baseline import Baseline
import torch
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
from utils import compute_cat_iou, compute_mACC




def fastTgcnEasyPredict(inDir, outDir, modelPath):
    
    
    
    #bring in new observations to be predicted on via the data loader
    inSet = plydataset(path = inDir, arch = "u", mode = 'test', model = 'meshsegnet')
    loader = DataLoader(inSet, batch_size=1, shuffle=True, num_workers=8)
    
    #set up the model the same way that it was set up in the training process
    segModel = Baseline(in_channels=12, output_channels=17)
    
    #load in the model weights from the desired iteration of the training process
    state = torch.load(modelPath, map_location="cuda", weights_only=True)
    segModel.load_state_dict(state)
    
    #transfer model to gpu
    segModel = segModel.cuda()
    
    #put the model in training mode, usually, when doing prediction, the model
    #should be in evaluation mode, however, the segmentation on the test data within
    #the training process was done with the model in training mode and results were
    #very good. Previously, i tried using evaluation mode and got very poor results
    #so training mode it is
    segModel.train()
    #confirm it is in training mode
    print("Model in training mode?", segModel.training)
    
    
    
    
    #####################################
    #code from this point is directly from the training loop. The remainder of 
    #the code is a bit messy but it works. Instead of spending time trying to clean
    #it up, I am going to keep it as-is and return later to make things a bit less
    #held together with popsicle sticks and gum. So if you are not me, please forgive
    #the un-optimal coding below
    ####################################
    
    #arguements
    model = segModel
    #loader = loader #loader already named properly above
    arch = "u"
    plyPath=outDir
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
    # print(plyPathStr)
    # print(plyPathStr1)
    # print(plyPathStr2)
    
    print("a")
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
    



