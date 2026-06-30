#!/bin/bash
#$ -l ngpus=3
#$ -pe smp 32
#$ -o outputFiles/$JOB_ID_warmstartOut.o
#$ -e outputFiles/$JOB_ID_warmstartError.e



apptainer exec ../../../containers/pytorch2.sif python trainWarmstartRunner.py
