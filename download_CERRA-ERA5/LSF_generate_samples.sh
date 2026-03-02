#!/bin/bash
# embedded options to bsub - start with #BSUB
# -- our name ---
#BSUB -J CERRA-ERA5_sample_generation
# -- choose queue --
#BSUB -q hpc
# -- specify that we need 4GB of memory per core/slot --
# so when asking for 4 cores, we are really asking for 4*4GB=16GB of memory 
# for this job. 
#BSUB -R "rusage[mem=1GB]"
# -- Notify me by email when execution begins --
#BSUB -B
# -- Notify me by email when execution ends   --
#BSUB -N
# -- email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s232493@dtu.dk
# -- Output File --
#BSUB -o /zhome/98/d/202490/Documents/Thesis/ClimateDiffuse/.lsf_jobs/Output_%J.out
# -- Error File --
#BSUB -e /zhome/98/d/202490/Documents/Thesis/ClimateDiffuse/.lsf_jobs/Output_%J.err
# -- estimated wall clock time (execution time): hh:mm -- 
#BSUB -W 00:15
# -- Number of cores requested -- 
#BSUB -n 1 
# -- Specify the distribution of the cores: on a single node --
#BSUB -R "span[hosts=1]"
# -- end of LSF options -- 

# loads automatically also numpy and python3 and underlying dependencies for our python 3.11.7
# module load pandas/2.1.3-python-3.11.7

# in case you have created a virtual environment,
# activate it first:
source /zhome/98/d/202490/Documents/Thesis/clim-dif/bin/activate

# use this for just piping everything into a file, 
# the program knows then, that it's outputting to a file
# and not to a screen, and also combine stdout&stderr
bash /zhome/98/d/202490/Documents/Thesis/ClimateDiffuse/download_CERRA-ERA5/generate_samples.sh > /zhome/98/d/202490/Documents/Thesis/ClimateDiffuse/.lsf_jobs/joboutput_$LSB_JOBID.out 2>&1
