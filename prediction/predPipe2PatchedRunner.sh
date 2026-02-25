#!/bin/bash
#$ -l ngpus=3
#$ -pe smp 16
#$ -o outputFiles/$JOB_ID_predOutPatched.o
#$ -e outputFiles/$JOB_ID_predErrorPatched.e



apptainer exec ../../../containers/pytorch2.sif python predPipe2Patched.py
