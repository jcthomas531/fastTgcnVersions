#!/bin/bash
#$ -l ngpus=3
#$ -pe smp 64
#$ -o outputFiles/$JOB_ID_fastTgcnEasyOut.o
#$ -e outputFiles/$JOB_ID_fastTgcnEasyError.e



apptainer exec ../../../containers/pytorch2.sif python trainRunner.py
