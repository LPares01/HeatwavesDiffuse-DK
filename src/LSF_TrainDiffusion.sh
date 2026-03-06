#!/bin/sh
### General options
# -- our name ---
#BSUB -J climdif_train_diffusion
### -- specify queue --
#BSUB -q gpuv100
### -- ask for number of cores (default: 4) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request X GB of system-memory
#BSUB -R "rusage[mem=8GB]"
### we want to have this on a single node
#BSUB -R "span[hosts=1]"
### for requesting a GPU with 32GB (only available in the gpuv100 queue) 
#BSUB -R "select[gpu32gb]"
### -- set the email address --
##BSUB -u s232493@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
# -- Output File --
#BSUB -o /zhome/98/d/202490/Documents/Thesis/ClimateDiffuse/.lsf_jobs/Output_%J.out
# -- Error File --
#BSUB -e /zhome/98/d/202490/Documents/Thesis/ClimateDiffuse/.lsf_jobs/Output_%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/11.6

/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

# in case you have created a virtual environment,
# activate it first:
source /path/to/your/venv/bin/activate

# use this for just piping everything into a file, 
# the program knows then, that it's outputting to a file
# and not to a screen, and also combine stdout&stderr
python ./TrainDiffusion.py > ../.lsf_jobs/joboutput_$LSB_JOBID.out 2>&1
