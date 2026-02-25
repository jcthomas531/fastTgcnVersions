# compare_pipelines.py
# Runs a single batch through multiple inference variants and compares metrics.
# Author: (you)

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

# --- repo-local imports (must match your project layout) ---
from dataloader import plydataset               # from your repo
from Baseline import Baseline                   # from your repo
from utils import compute_cat_iou, compute_mACC # metric helpers from your repo

# --------------------------
# Utilities
# --------------------------
def set_all_seeds(seed: int = 1):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def tensor_stats(name, t, max_channels=12):
    # t expected shapes: 
    #   coordinate: [B, C, N]
    #   idx_face:   [B, N] or [B, N, ...]
    with torch.no_grad():
        print(f"\n[{name}] dtype={t.dtype}, shape={tuple(t.shape)}, device={t.device}")
        if t.dim() == 3:   # e.g., [B, C, N]
            B, C, N = t.shape
            c = min(C, max_channels)
            means = t[:, :c, :].float().mean(dim=(0, 2)).detach().cpu().numpy()
            stds  = t[:, :c, :].float().std(dim=(0, 2)).detach().cpu().numpy()
            print(f"[{name}] first {c} channel means: {np.round(means, 4)}")
            print(f"[{name}] first {c} channel stds : {np.round(stds, 4)}")
        else:
            # For index-like tensors
            mn = float(t.min())
            mx = float(t.max())
            mean = float(t.float().mean())
            std  = float(t.float().std())
            print(f"[{name}] min/max={mn:.3f}/{mx:.3f}, mean={mean:.3f}, std={std:.3f}")

def forward_and_score(model, coordinate, idx_face, label_face, num_classes=17, mode="eval"):
    """
    Runs a forward pass with specific mode (eval/train) and returns metrics.
    label_face expected shape: [B, N] (class ids)
    coordinate shape: [B, C, N]
    idx_face same shape/type as produced by your dataloader
    """
    if mode == "eval":
        model.eval()
    else:
        model.train()

    with torch.no_grad():
        pred = model(coordinate, idx_face)      # [B, N, num_classes] or [B*N, num_classes] depending on model
        # metrics in your repo expect pred as [B, N, C]
        if pred.dim() == 3 and pred.shape[-1] == num_classes:
            pred_logits = pred
        else:
            # If model returns [B, N, C] but transposed differently, adjust if needed.
            # Common fallback: [B, N, C]
            pred_logits = pred

        # Compute metrics using the same helpers the repo uses (utils.py)
        # test_semseg pipeline does:
        #   mdice += dice_loss(pred.max(dim=-1)[0], label_face)
        #   iou_table, _ = compute_cat_iou(pred, label_face, iou_table)
        # then flattens to compute accuracy/mACC
        # We'll compute IoU + accuracy + mACC for this single batch.

        # IoU
        iou_table = np.zeros((num_classes, 3))
        iou_table, _ = compute_cat_iou(pred_logits, label_face, iou_table)
        # finalize mean IoU across present classes
        with np.errstate(divide='ignore', invalid='ignore'):
            iou_table[:, 2] = np.where(iou_table[:, 1] > 0, iou_table[:, 0] / iou_table[:, 1], 0.0)
        mIoU = float(np.nanmean(iou_table[:, 2]))

        # Flatten for accuracy/mACC like test_semseg
        B, N, C = pred_logits.shape
        flat_pred = pred_logits.contiguous().view(-1, C)
        flat_label = label_face.view(-1)

        pred_choice = flat_pred.data.max(1)[1]
        correct = pred_choice.eq(flat_label.data).cpu().sum()
        acc = float(correct.item()) / (B * N)

        macc = float(compute_mACC(pred_choice, flat_label).cpu().data.numpy())

        # distribution (optional, for sanity)
        hist = torch.bincount(pred_choice, minlength=num_classes).cpu().numpy()
        return {
            "acc": acc,
            "mIoU": mIoU,
            "mAcc": macc,
            "pred_hist": hist,
        }

def main():
    parser = argparse.ArgumentParser(description="Compare inference pipelines on a single batch.")
    parser.add_argument("--data_path", required=True, type=str, help="Folder with test PLYs (same one used during the good run).")
    parser.add_argument("--arch", default="u", choices=["l","u"], help="Arch flag used by your dataloader.")
    parser.add_argument("--ckpt", required=True, type=str, help="Path to model .pth checkpoint.")
    parser.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    parser.add_argument("--batch_index", default=0, type=int, help="Which batch index to test (0-based).")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--batch_size", default=1, type=int)  # keep 1 to mimic repo
    parser.add_argument("--num_classes", default=17, type=int)
    args = parser.parse_args()

    set_all_seeds(1)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    # --------------------------
    # Loader (match repo defaults)
    # --------------------------
    dataset = plydataset(path=args.data_path, arch=args.arch, mode="test", model="meshsegnet")
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Fetch the chosen batch deterministically
    batch = None
    for i, data in enumerate(loader):
        if i == args.batch_index:
            batch = data
            break
    if batch is None:
        raise IndexError(f"Batch index {args.batch_index} out of range for dataset of length {len(dataset)}")

    # Unpack exactly like repo codepaths do:
    # utils.test_semseg: (index, points, label_face, label_face_onehot, name, raw_points_face, idx_face)
    # train.py:          (_, points_face, label_face, label_face_onehot, name, _, index_face)
    index, points, label_face, label_face_onehot, name, raw_points_face, idx_face = batch

    # Shapes & label prep (match test_semseg)
    # test_semseg does: label_face = label_face[:, :, 0]
    label_face = label_face[:, :, 0].long()
    B, N, _ = points.size()

    # Build both coordinate variants
    # Variant A (test_semseg-style naming): uses "points"
    coord_testsemseg = points.transpose(2, 1).float()

    # Variant B (training-loop-style naming): uses "points_face"
    # In train.py they name the 2nd item "points_face"; numerically it should be the same tensor
    # but we still test both paths explicitly.
    points_face = points  # alias to mirror naming in train.py
    coord_trainloop = points_face.transpose(2, 1).float()

    # idx_face variants
    idx_face_float = idx_face.float()
    idx_face_long  = idx_face.long()

    # Move everything to device
    coord_testsemseg = coord_testsemseg.to(device)
    coord_trainloop  = coord_trainloop.to(device)
    idx_face_float   = idx_face_float.to(device)
    idx_face_long    = idx_face_long.to(device)
    label_face       = label_face.to(device)

    # Print sanity stats
    print("===== INPUT STATS =====")
    tensor_stats("coord_testsemseg", coord_testsemseg)
    tensor_stats("coord_trainloop ", coord_trainloop)
    tensor_stats("idx_face_float  ", idx_face_float)
    tensor_stats("idx_face_long   ", idx_face_long)
    print(f"\n[labels] shape={tuple(label_face.shape)}, dtype={label_face.dtype}, unique={torch.unique(label_face).cpu().tolist()}")

    # --------------------------
    # Model + checkpoint
    # --------------------------
    model = Baseline(in_channels=12, output_channels=args.num_classes).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state, strict=True)

    # --------------------------
    # Run 8 combinations and report
    # --------------------------
    combos = [
        ("testsemseg_coord", "float_idx", "eval",  coord_testsemseg, idx_face_float),
        ("testsemseg_coord", "float_idx", "train", coord_testsemseg, idx_face_float),
        ("testsemseg_coord", "long_idx",  "eval",  coord_testsemseg, idx_face_long),
        ("testsemseg_coord", "long_idx",  "train", coord_testsemseg, idx_face_long),
        ("trainloop_coord",  "float_idx", "eval",  coord_trainloop,  idx_face_float),
        ("trainloop_coord",  "float_idx", "train", coord_trainloop,  idx_face_float),
        ("trainloop_coord",  "long_idx",  "eval",  coord_trainloop,  idx_face_long),
        ("trainloop_coord",  "long_idx",  "train", coord_trainloop,  idx_face_long),
    ]

    results = []
    for coord_name, idx_name, mode, coord, idx in combos:
        out = forward_and_score(model, coord, idx, label_face, num_classes=args.num_classes, mode=mode)
        results.append((coord_name, idx_name, mode, out))
        print(f"\n=== {coord_name} | {idx_name} | mode={mode} ===")
        print(f"acc={out['acc']:.4f}  mIoU={out['mIoU']:.4f}  mAcc={out['mAcc']:.4f}")
        print(f"pred_hist (first 10 classes): {out['pred_hist'][:10]}")

    # Compare predictions between key pairs (optional: you can flesh this out)
    print("\n===== SUMMARY TABLE =====")
    for coord_name, idx_name, mode, out in results:
        print(f"{coord_name:16s} | {idx_name:9s} | {mode:5s} -> acc={out['acc']:.4f}, mIoU={out['mIoU']:.4f}, mAcc={out['mAcc']:.4f}")

if __name__ == "__main__":
    main()