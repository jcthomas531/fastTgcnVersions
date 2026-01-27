#for testing
# import os
# os.chdir("H:\\schoolFiles\\dissertation\\fastTgcnVersions\\fastTgcnEasy")
# os.getcwd()


from dataloader import plydataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import numpy as np
import os
# from tensorboardX import SummaryWriter
from torch.autograd import Variable
from pathlib import Path
import torch.nn.functional as F
import datetime
import logging
from sklearn.model_selection import StratifiedKFold
from utils import test_semseg
from loss import IoULoss, DiceLoss
#from TSGCNet import TSGCNet
#from TestModel import TestModel
#from PointNet import PointNetDenseCls
#from PointNetplus import PointNet2
#from MeshSegNet import MeshSegNet
from Baseline import Baseline
#from ablation import ablation
#from OurMethod import SGNet
#from pct import PointTransformerSeg
import random






#this if __name__ == "__main__" statement here prevents the code from being run
#if this function is imported into another script
#see: https://www.reddit.com/r/learnpython/comments/eb57p0/what_is_the_point_of_name_main_in_python_programs/






#note 1
#the original code is set up so that when the scirpt is ran, there are no outputs
#to the environment, just saved items. I am hopping that things should still save 
#in the correct locations even when inside a function

#note 2
#set up to use arch as an arguement, down stream functions that vary by arch were 
#also updated to use this dynamically


#next step
#if this can run fine, it would be nice to be able have the outputs labeled with
#the arch as well


def fastTgcnEasy(arch, testPath, trainPath, batch_size = 1, k = 32,
                 numWorkers = 8, epochs = 301):
    
    
    ###########################################################################
    #timing
    ###########################################################################
    startTime = time.time()
    
    
    ###########################################################################
    #checking function arguements
    ###########################################################################
    if arch not in {"l", "u"}:
        raise ValueError("Arguement Arch must be either 'l' or 'u'")
        
        
    
    
    ###########################################################################
    #set up
    ###########################################################################
    #require 3 GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    """-------------------------- parameters --------------------------------------"""
    #below is from original code, now these are set up via arguments
    # batch_size = 1
    # k = 32
    
    """--------------------------- create Folder ----------------------------------"""
    current_time = str(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
    #
    #to include everthing in
    modelOutDir = Path("./modelOutputs")
    modelOutDir.mkdir(exist_ok=True)
    
    #subfolder for this specific run
    runDir = Path(str(modelOutDir) + '/run' + current_time)
    runDir.mkdir(exist_ok=True)
    
    #using the old object names but with new locations
    #a little messy and repetative but requires least rewriting
    experiment_dir = Path(str(runDir) + '/checkpointsAndLogs/')
    experiment_dir.mkdir(exist_ok=True)
    pred_dir = Path(str(runDir) + '/pred_global/')
    pred_dir.mkdir(exist_ok=True)
    file_dir = experiment_dir
    file_dir.mkdir(exist_ok=True)
    log_dir, checkpoints = file_dir.joinpath('logs/'), file_dir.joinpath('checkpoints')
    log_dir.mkdir(exist_ok=True)
    checkpoints.mkdir(exist_ok=True)
    
    #old
    # experiment_dir = Path('./experiment/')
    # experiment_dir.mkdir(exist_ok=True)
    # pred_dir = Path('./pred_global/')
    # pred_dir.mkdir(exist_ok=True)
    # file_dir = Path(str(experiment_dir) + '/test-1')
    # file_dir.mkdir(exist_ok=True)
    # log_dir, checkpoints = file_dir.joinpath('logs/'), file_dir.joinpath('checkpoints')
    # log_dir.mkdir(exist_ok=True)
    # checkpoints.mkdir(exist_ok=True)

    formatter = logging.Formatter('%(name)s - %(message)s')
    logger = logging.getLogger("all")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(str(log_dir) + '/log.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # writer = SummaryWriter(file_dir.joinpath('tensorboard'))
    
    
    
    #setting overall seed and seed for workers
    torch.cuda.manual_seed(1)
    def worker_init_fn(worker_id):
        random.seed(1 + worker_id)
        
        
        
        
    ###########################################################################
    #bringing in data
    ###########################################################################
    #questions: why is batch size allowed to differ between train and test loader
    #the original code had batch_size set to 1, could the hard coded 1 just have 
    #been left in error?
    """-------------------------------- Dataloader --------------------------------"""
    if arch == "l":
        train_dataset_4 = plydataset(path = trainPath, arch = arch, mode = 'train', model = 'meshsegnet')
        train_loader_4 = DataLoader(train_dataset_4, batch_size=batch_size, shuffle=True, num_workers=numWorkers, worker_init_fn=worker_init_fn)
        test_dataset_4 = plydataset(path = testPath, arch = arch, mode = 'test', model = 'meshsegnet')
        test_loader_4 = DataLoader(test_dataset_4, batch_size=1, shuffle=True, num_workers=numWorkers)
    elif arch == "u":
        train_dataset_4 = plydataset(path = trainPath, arch = arch, mode = 'train', model = 'meshsegnet')
        train_loader_4 = DataLoader(train_dataset_4, batch_size=batch_size, shuffle=True, num_workers=numWorkers,worker_init_fn=worker_init_fn)
        test_dataset_4 = plydataset(path = testPath, arch = arch, mode = 'test', model = 'meshsegnet')
        test_loader_4 = DataLoader(test_dataset_4, batch_size=1, shuffle=True, num_workers=numWorkers)
        
    #THIS BRINGS IN THE RIGHT DATA BUT WHAT ABOUT THE LABELING FROM FUNCTIONS IN dataloader.py
    
    
    
    ###########################################################################
    #Build Network and optimizer
    ###########################################################################
    #note 1
    #output_channels is very interesting. there are a max number of 16 
    #teeth on each arch and with the gums that makes 17 things to categorize.
    #one thing we have discussed is the limits of the model when there is the wrong
    #amount of teeth. It seems to want to hit the 17 categories. Perhaps this could
    #be imporved by supplying the number of teeth (if this is something that is
    #collected) to the model. I am not sure if this model constrains each possible
    #output category to be used, but that would be something good to look into. 
    #our other thought for addressing this was building in information into the
    #loss function
    #note 2
    #some of these hyperparameters my be good as arguements, worth looking at 
    #but for this first past, I will keep like this
    #note 3
    #i dont know much about in_channels, can this be vaired or is it set?
    """--------------------------- Build Network and optimizer----------------------"""
    model = Baseline(in_channels=12, output_channels=17)
    model.cuda()
    optimizer = torch.optim.Adam(
    model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    
    
    
    ###########################################################################
    #train
    ###########################################################################
    #note 1
    #there is so much commented out here that i do not understand. were these things
    #from prior iterations/things that they were trying or were these options?
    #perhaps the paper will shine some light on this
    #i believe I will need a better knowledge base in pytorch and deep learning
    #in order to understand this bit
    #note 2
    #find the spot that returns the final error rates in this section and return
    #them as function returns as well
    """------------------------------------- train --------------------------------"""
    logger.info("------------------train------------------")
    best_acc = 0
    best_miou = 0
    best_macc = 0
    LEARNING_RATE_CLIP = 1e-5
    his_loss = []
    his_smotth = []
    class_weights = torch.ones(15).cuda()
    iou_loss = IoULoss()
    dice_loss = DiceLoss()
    # iou_label = torch.ones((1, 17)).float().cuda()
    for epoch in range(0, epochs):
        train_loader = train_loader_4
        test_loader = test_loader_4
        scheduler.step()
        lr = max(optimizer.param_groups[0]['lr'], LEARNING_RATE_CLIP)
        optimizer.param_groups[0]['lr'] = lr

        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
            _, points_face, label_face, label_face_onehot, name, _, index_face = data
            coordinate = points_face.transpose(2,1)
            # coordinate, label_face = Variable(coordinate.float()), Variable(label_face.long())
            coordinate, label_face, index_face = Variable(coordinate.float()), Variable(label_face.long()), Variable(index_face.float())
            label_face_onehot = Variable(label_face_onehot)
            # coordinate, label_face, label_face_onehot = coordinate.cuda(), label_face.cuda(), label_face_onehot.cuda()
            coordinate, label_face, label_face_onehot, index_face = coordinate.cuda(), label_face.cuda(), label_face_onehot.cuda(), index_face.cuda()
            optimizer.zero_grad()


            # iou_tabel = torch.zeros((17, 3)).float().cuda()
            # print(iou_tabel.shape)
            pred = model(coordinate, index_face)
            label_face = label_face.view(-1, 1)[:, 0]
            pred = pred.contiguous().view(-1, 17)
            # pred_ = pred.max(dim=-1)[0]
            # print(pred.shape)
            # print(pred_.shape)

            
            ###################################################################
            #loss
            loss1 = F.nll_loss(pred, label_face)
            # loss2 = dice_loss(pred.max(dim=-1)[0], label_face)
            # print(loss2)
            loss = loss1
            # loss = F.nll_loss(pred, label_face) + F.l1_loss(iou, iou_label)
            # loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            his_loss.append(loss.cpu().data.numpy())
            ###################################################################
            
            
        #this part is just purely outputting stuff
        if epoch % 10 == 0:
            print('Learning rate: %f' % (lr))
            print("loss: %f" % (np.mean(his_loss)))
            # writer.add_scalar("loss", np.mean(his_loss), epoch)
            metrics, mIoU, cat_iou, mAcc, throwAway = test_semseg(model = model, loader = test_loader, arch = arch, plyPath=pred_dir,
                                                                  num_classes=17, generate_ply=True)
            print("Epoch %d, accuracy= %f, mIoU= %f, mACC= %f" % (epoch, metrics['accuracy'], mIoU, mAcc))
            logger.info("Epoch: %d, accuracy= %f, mIoU= %f, mACC= %f loss= %f" % (epoch, metrics['accuracy'], mIoU, mAcc, np.mean(his_loss)))
            # writer.add_scalar("accuracy", metrics['accuracy'], epoch)
            print("best accuracy: %f best mIoU :%f, mACC: %f" % (best_acc, best_miou, best_macc))
            if ((metrics['accuracy'] > best_acc) or (mIoU > best_miou) or (mAcc > best_macc)):
                if metrics['accuracy'] > best_acc:
                    best_acc = metrics['accuracy']
                if mIoU > best_miou:
                    best_miou = mIoU
                if mAcc > best_macc:
                    best_macc = mAcc
                print("best accuracy: %f best mIoU :%f, mACC: %f" % (best_acc, best_miou, mAcc))
                print(cat_iou)
                torch.save(model.state_dict(), '%s/coordinate_%d_%f.pth' % (checkpoints, epoch, best_acc))
                best_pth = '%s/coordinate_%d_%f.pth' % (checkpoints, epoch, best_acc)
                logger.info(cat_iou)
            his_loss.clear()
            # writer.close()
            
        if epoch == (epochs - 1):
            #above needs some work as i think it is outputting the final plys not the best plys
            #ideally we will output the best plys, i will need to spend more time 
            #with the above code to ensure that the output model and saved plys
            #are the best, just not the last, but this is sufficient for the time beings
            
            #get final time and format it
            endTime = time.time()
            totalTime = endTime - startTime
            timeHours = int(totalTime // 3600)
            timeMinutes = int((totalTime % 3600) // 60)
            timeSeconds = int(totalTime % 60)
            
            #output final stats and time
            print("\n############### FINISHED ###############\n")
            print("\n")
            print(f"Arch: {arch}")
            print("\n")
            print(f"Best Accuracy: {best_acc:.4f}")
            print(f"Best Mean IoU: {best_miou:.4f}")
            print(f"Best Mean Class Accuracy: {best_macc:.4f}") 
            print("\n")
            print(f"Current Epoch: {epoch:.4f}")
            print(f"Training Epochs: {epochs:.4f}") 
            print("Best Epoch: ADD LATER")
            print("Model Output Path: ADD LATER")
            print("Ply Output Path: ADD LATER")
            print(f"Total Training Time: {timeHours}h {timeMinutes}m {timeSeconds}s")
            print("\n########################################\n")
            
        
                
                

                
                
                
            
    #IN THE ORIGINAL CODE THERE WAS A COMMENTED OUT SECTION AFTER THIS WITH 200
    #EPOCHS. NOT SURE WHY THIS WAS LEFT IN, AGAIN, WAS THIS AN OLDER VERSION/SOMETHING
    #THAT THEY WERE WORKING ON OR WAS IT AN OPTION. AGAIN, I WILL NEED BETTER
    #PYTORCH KNOWLEDGE AND DEEP LEARNING KNOWLEDGE TO FIGURE THAT OUT
    



















