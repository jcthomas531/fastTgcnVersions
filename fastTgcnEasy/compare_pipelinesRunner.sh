#!/bin/bash
#$ -l ngpus=3
#$ -pe smp 16
#$ -o outputFiles/$JOB_ID_predOut.o
#$ -e outputFiles/$JOB_ID_predError.e



apptainer exec ../../../containers/pytorch2.sif python compare_pipelines.py \
  --data_path "/Shared/gb_lss/Thomas/IOSSegData/clean/trainCleanU" \
  --arch u \
  --ckpt "/Users/jthomas48/dissModels/fastTgcnVersions/fastTgcnEasy/modelOutputs/2026_01_27 full upper/checkpointsAndLogs/checkpoints/coordinate_220_0.917194.pth" \
  --device cuda \
  --batch_index 0 \
  --num_workers 0 \
  --batch_size 1 \
  --num_classes 17

