#!/bin/bash
#$ -l ngpus=3
#$ -pe smp 64
#$ -o outputFiles/$JOB_ID_fastTgcnLossOut.o
#$ -e outputFiles/$JOB_ID_fastTgcnLossError.e



apptainer exec ../../../containers/pytorch2.sif python trainRunner.py
