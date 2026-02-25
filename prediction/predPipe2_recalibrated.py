# predPipe2_recalibrated.py
# Purpose: Replicate your original predPipe2.py but add BN recalibration (Option B)
#          and an option to run in train-mode inference (Option A) to match
#          training-time test_semseg behavior.
#
# How to use (examples):
#   python predPipe2_recalibrated.py \
#       --data_path "/Shared/gb_lss/Thomas/iowaRme/test1" \
#       --arch u \
#       --ckpt "/Users/jthomas48/dissModels/fastTgcnVersions/fastTgcnEasy/modelOutputs/2026_01_27 full upper/checkpointsAndLogs/checkpoints/coordinate_220_0.917194.pth" \
#       --ply_out "/Shared/gb_lss/Thomas/iowaRme/test1Pred/" \
#       --use_bn_recal 1 --bn_iters 50 --generate_ply 1
#
#   # Or to force train-mode inference (no recalibration):
#   python predPipe2_recalibrated.py --use_bn_recal 0 --train_mode_infer 1

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict

# Allow imports from your repo folder (adjust if needed)
sys.path.append("/Users/jthomas48/dissModels/fastTgcnVersions/fastTgcnEasy/")

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

# Repo-local imports
from dataloader import plydataset, generate_plyfile
from Baseline import Baseline
from loss import IoULoss, DiceLoss
from utils import compute_cat_iou, compute_mACC


def set_all_seeds(seed: int = 1):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(data_path: str, arch: str, batch_size: int = 1, num_workers: int = 1):
    dataset = plydataset(path=data_path, arch=arch, mode='test', model='meshsegnet')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader


def recalibrate_bn(model, loader, iters: int = 50, device: str = 'cuda'):
    """Recompute BN running mean/var on several mini-batches, no weight updates."""
    model.train()
    count = 0
    with torch.no_grad():
        for batch_id, (index, points, label_face, label_face_onehot, name, raw_points_face, idx_face) in enumerate(loader):
            coordinate = points.transpose(2,1).float().to(device)
            idx_face_t = idx_face.float().to(device)
            _ = model(coordinate, idx_face_t)
            count += 1
            if count >= iters:
                break
    # Don't switch to eval here; the caller decides the mode.


def run_inference(model,
                  loader,
                  arch: str,
                  ply_out: str,
                  num_classes: int = 17,
                  generate_ply: bool = True,
                  device: str = 'cuda'):
    """Mirror utils.test_semseg() logic but as a standalone function, with optional PLY write."""
    iou_tabel = np.zeros((num_classes, 3))
    metrics = defaultdict(lambda: list())
    dice_loss = DiceLoss()
    mdice = 0.0
    macc = 0.0

    # Prepare output path (consistent with original behavior but safer)
    plyPathStr = str(ply_out)
    plyPathStr1 = "./" + plyPathStr.replace("\\\\", "/")
    plyPathStr2 = plyPathStr.replace("\\\\", "/") + "/%s"

    Path(plyPathStr1).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_id, (index, points, label_face, label_face_onehot, name, raw_points_face, idx_face) in \
                tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
            batchsize, num_point, _ = points.size()
            point_face = raw_points_face[0].numpy()
            index_face = index[0].numpy()

            # Build inputs just like utils.test_semseg
            coordinate = points.transpose(2, 1).float().to(device)
            label_face = label_face[:, :, 0].long().to(device)
            centre = points[:, :, 9:12].float().to(device)   # not used directly but kept for parity
            idx_face_t = idx_face.float().to(device)

            pred = model(coordinate, idx_face_t)

            mdice += float(dice_loss(pred.max(dim=-1)[0], label_face))
            iou_tabel, _ = compute_cat_iou(pred, label_face, iou_tabel)

            pred_flat = pred.contiguous().view(-1, num_classes)
            label_flat = label_face.view(-1)
            pred_choice = pred_flat.data.max(1)[1]

            macc += float(compute_mACC(pred_choice, label_flat).cpu().data.numpy())
            correct = pred_choice.eq(label_flat.data).cpu().sum()
            metrics['accuracy'].append(correct.item() / (batchsize * num_point))

            # Optional PLY writing
            if generate_ply:
                out_labels = pred_choice.cpu().reshape(pred_choice.shape[0], 1).numpy()
                generate_plyfile(index_face=index_face,
                                 point_face=point_face,
                                 label_face=out_labels,
                                 arch=arch,
                                 path=(plyPathStr2) % name)

    # Finalize metrics to match utils.test_semseg
    iou_tabel[:, 2] = np.divide(iou_tabel[:, 0], iou_tabel[:, 1], out=np.zeros_like(iou_tabel[:, 0]), where=iou_tabel[:, 1] > 0)
    metrics['accuracy'] = float(np.mean(metrics['accuracy'])) if len(metrics['accuracy']) > 0 else 0.0
    mIoU = float(np.nanmean(iou_tabel[:, 2]))

    return metrics, mIoU, macc, mdice, iou_tabel


def main():
    parser = argparse.ArgumentParser(description="Inference with BN recalibration or train-mode.")
    parser.add_argument('--data_path', type=str, default='/Shared/gb_lss/Thomas/iowaRme/test1')
    parser.add_argument('--arch', type=str, choices=['l', 'u'], default='u')
    parser.add_argument('--ckpt', type=str, required=True,
                        default='/Users/jthomas48/dissModels/fastTgcnVersions/fastTgcnEasy/modelOutputs/2026_01_27 full upper/checkpointsAndLogs/checkpoints/coordinate_220_0.917194.pth')
    parser.add_argument('--ply_out', type=str, default='/Shared/gb_lss/Thomas/iowaRme/test1Pred/')

    parser.add_argument('--use_bn_recal', type=int, default=1, help='1=recompute BN running stats, then eval(); 0=skip')
    parser.add_argument('--bn_iters', type=int, default=50, help='how many mini-batches to use to recalibrate BN')
    parser.add_argument('--train_mode_infer', type=int, default=0, help='if 1, force train-mode inference (ignores use_bn_recal)')

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=17)
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--generate_ply', type=int, default=1)

    args = parser.parse_args()

    set_all_seeds(args.seed)

    device = 'cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu'

    # Build data loader
    loader = build_loader(args.data_path, args.arch, batch_size=args.batch_size, num_workers=args.num_workers)

    # Build and load model
    model = Baseline(in_channels=12, output_channels=args.num_classes)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model = model.to(device)

    # ---- Choose inference strategy ----
    if args.train_mode_infer:
        # Option A: train-mode inference (matches training-time test_semseg behavior)
        model.train()
        print("[INFO] Using TRAIN-MODE inference (BatchNorm uses batch stats).")
    else:
        if args.use_bn_recal:
            # Option B: BN recalibration then eval-mode
            print(f"[INFO] Recalibrating BN running stats for {args.bn_iters} mini-batches...")
            recalibrate_bn(model, loader, iters=args.bn_iters, device=device)
            model.eval()
            print("[INFO] Switched to EVAL mode after BN recalibration.")
        else:
            # Plain eval (not recommended given your results)
            model.eval()
            print("[INFO] Using plain EVAL mode (no BN recalibration).")

    print("model.training =", model.training)

    # Run inference over the dataset
    metrics, mIoU, macc, mdice, iou_table = run_inference(
        model=model,
        loader=loader,
        arch=args.arch,
        ply_out=args.ply_out,
        num_classes=args.num_classes,
        generate_ply=bool(args.generate_ply),
        device=device
    )

    print("\n============= RESULTS =============")
    print(f"Accuracy (mean over batches): {metrics['accuracy']:.4f}")
    print(f"mIoU: {mIoU:.4f}")
    print(f"mAcc: {macc:.4f}")
    print(f"Dice (sum proxy): {mdice:.4f}")
    print("==================================\n")


if __name__ == '__main__':
    main()
