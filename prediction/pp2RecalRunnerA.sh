#!/bin/bash
#$ -l ngpus=3
#$ -pe smp 16
#$ -o outputFiles/$JOB_ID_predOut_recalA.o
#$ -e outputFiles/$JOB_ID_predError_recalA.e



apptainer exec ../../../containers/pytorch2.sif python predPipe2_recalibrated.py \
  --data_path "/Shared/gb_lss/Thomas/iowaRme/test1" \
  --arch u \
  --ckpt "/Users/jthomas48/dissModels/fastTgcnVersions/fastTgcnEasy/modelOutputs/2026_01_27 full upper/checkpointsAndLogs/checkpoints/coordinate_220_0.917194.pth" \
  --ply_out "/Shared/gb_lss/Thomas/iowaRme/test1PredA/" \
  --use_bn_recal 1 \
  --bn_iters 50 \
  --generate_ply 1 \
  --device cuda


