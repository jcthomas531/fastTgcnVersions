# General

Modified version of original fastTgcn model. The code as written in the original fastTgcn repository (https://github.com/MIVRC/Fast-TGCN) does not work out of the box. This is a minimally updated version that fixes these small issues and gets the model running according to how it is described in the readme of the original repository. Details of these changes listed below. All other versions of fastTgcn in this repository are built up from fastTgcnBase.

Run via:

```shell
python train.py
```


# Changes from original

1. There are a number of python packages that are loaded in by the code that are not used, some of which are custom functions and repositories that cannot be gotten with pip install. I have commented out these imports in train.py:
    * from tensorboardX import SummaryWriter (should be commented out already) 
    * from TSGCNet import TSGCNet 
    * from TestModel import TestModel 
    * from PointNet import PointNetDenseCls 
    * from PointNetplus import PointNet2 
    * from MeshSegNet import MeshSegNet
    * from ablation import ablation 
    * from OurMethod import SGNet 
    * from pct import PointTransformerSeg 
1. In the train.py file in the dataLoader section, they use the function plydataset() that was created in the dataloader.py file. The first argument to this function is a file path. They have supplied the paths "data/train-L" and "data/test-L" which contradicts how they describe to set up the directory. As described by the readme, I have the data sitting in "data/train" and data/test". I have both the upper and lower data together. Because their code is set up with separate loadings for the upper and lower arches, it makes me think this model should be trained with upper and lower arches separately. This would make some of sense because the numbering system on upper and lower teeth are different. I have changed my directory set up to be L and U serparate (data/test-L, data/train-L, data/test-U, data/train-U) as well as the data/test and data/train being retained. 
1. Assignement error from test_semseg() function. this function returns 5 items but there was only 4 alloted to the assignment, i have added a 5th called throwAway. Not totally sure what it is at this point but I am saving it just in case. in train.py file: 
    * **before**: metrics, mIoU, cat\_iou, mAcc = test\_semseg(model, test\_loader, num\_classes=17, generate_ply=True) 
    * **after**: metrics, mIoU, cat\_iou, mAcc, throwAway = test\_semseg(model, test\_loader, num\_classes=17, generate_ply=True) 
