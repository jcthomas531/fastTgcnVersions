import torch
from torch.utils.data import DataLoader
import os
print("cwd:", os.getcwd())
import sys
sys.path.append("/Users/jthomas48/dissModels/fastTgcnVersions/fastTgcnEasy")
print("sys.path:", sys.path)
from Baseline import Baseline
from dataloader import plydataset
from utilsPred import test_semseg


pthPath = "/Users/jthomas48/dissModels/fastTgcnVersions/fastTgcnEasy/modelOutputs/2026_01_27 full upper/checkpointsAndLogs/checkpoints"


#require 3 GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
device = "cuda"


# 1) Load model
model = Baseline(in_channels=12, output_channels=17).to(device)
state = torch.load(pthPath+"/coordinate_220_0.917194.pth", map_location=device)
# FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
#   state = torch.load(pthPath+"/coordinate_220_0.917194.pth", map_location=device)
#NOTE it is fine to use the weights_only=True bc this file is just weights only but
#if we had output the full model we would be in a different boat
model.load_state_dict(state)
model.eval()

# 2) Build dataset/loader pointing to folder of new PLYs
arch = "u" 
data_dir = "/Shared/gb_lss/Thomas/iowaRme/test1"
dataset = plydataset(path=data_dir, arch=arch, mode="test", model="meshsegnet")
loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

# 3) Run with test_semseg (will export PLYs to ./pred_global if generate_ply=True)
os.chdir("/Shared/gb_lss/Thomas/iowaRme/test1Pred/")
from pathlib import Path
outDir = "/Shared/gb_lss/Thomas/iowaRme/test1Pred/"  
Path(outDir).mkdir(exist_ok=True)
with torch.no_grad():
    metrics, mIoU, cat_iou, mAcc, _ = test_semseg(
        model=model,
        loader=loader,
        arch=arch,
        plyPath=outDir,
        num_classes=17,
        generate_ply=True
    )
#above wants to use a gpu but none is available (although one should be bc we
#are operateing in my conda environement that was set up to interface with the
#gpu), it i switch to the cpu with the above cpu shim, i run out of memory...
#a problem that needs to be solved
#could decrease the size of my ply file as this file is much more dense than the 
#ones that were in iosSeg
#copilot had some suggestions but we will cross this bridge later
#this would not be a problem if i could just put the rme data on argon but I am 
#not sure the permissions allow that, need to speak with grant about that
#ALSO i think my outDir is not going where i want, it is being created in the fastTgcnVersions/fastTgcnEasy/ area
    
    