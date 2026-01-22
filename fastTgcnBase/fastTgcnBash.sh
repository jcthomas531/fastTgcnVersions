#!/bin/bash
#$ -l ngpus=3
#$ -pe smp 64
#$ -o outputFiles/fastTgcnOut.o
#$ -e outputFiles/fastTgcnError.e



apptainer exec ../../../containers/pytorch2.sif python train.py
