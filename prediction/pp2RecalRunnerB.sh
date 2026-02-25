#!/bin/bash
#$ -l ngpus=3
#$ -pe smp 16
#$ -o outputFiles/$JOB_ID_predOut_recalB.o
#$ -e outputFiles/$JOB_ID_predError_recalB.e



apptainer exec ../../../containers/pytorch2.sif python predPipe2_recalibrated.py \
  --data_path "/Shared/gb_lss/Thomas/iowaRme/test1" \
  --arch u \
  --ckpt "/Users/jthomas48/dissModels/fastTgcnVersions/fastTgcnEasy/modelOutputs/2026_01_27 full upper/checkpointsAndLogs/checkpoints/coordinate_220_0.917194.pth" \
  --ply_out "/Shared/gb_lss/Thomas/iowaRme/test1PredB/" \
  --use_bn_recal 0 \
  --train_mode_infer 1 \
  --generate_ply 1 \
  --device cuda



