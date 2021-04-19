#!/bin/bash

#! project to charge
#SBATCH -A NLP-CDT-SL2-GPU

#! number of nodes and tasks
#SBATCH --nodes=1
#SBATCH --ntasks 1

#! max wallclock time
#SBATCH --time=36:00:00

#! name of job
#SBATCH --job-name=base-bt

#! logging
#SBATCH --output=job_%x_%A_%a.out
#SBATCH --error=job_%x_%A_%a.err

#! number of GPUs
#SBATCH --gres=gpu:4

#! CPUs per task
#SBATCH --cpus-per-task 8

#! spawn multiple jobs to keep training model
#SBATCH --array=1-20%1

#! mail alert at start, end, and abortion of execution
#SBATCH --mail-type=ALL

#SBATCH -p pascal

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')
#! ############################################################
#! Modify the settings below to specify the application's environment, location 
#! and launch method:

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel7/default-gpu              # REQUIRED - loads the basic environment

#! Insert additional module load commands after this line if needed:
module load cuda/10.1
module load cudnn/7.6_cuda-10.1

#! Full path to application executable: 
application="/home/cs-burc1/projects/diversity-bt/csd3-scripts/train-bt-models/train-baseline-bt-de2.sh"

#! define inputs to script
options=$1

#! Work directory (i.e. where the job will run):
workdir="$SLURM_SUBMIT_DIR"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory
                             # in which sbatch is run.

#! Are you using OpenMP (NB this is unrelated to OpenMPI)? If so increase this
#! safe value to no more than 12:
export OMP_NUM_THREADS=1

#! Number of MPI tasks to be started by the application per node and in total (do not change):
np=$[${numnodes}*${mpi_tasks_per_node}]

#! Choose this for a pure shared-memory OpenMP parallel program on a single node:
#! (OMP_NUM_THREADS threads will be created):
CMD="$application $options"

#! Choose this for a MPI code using OpenMPI:
#CMD="mpirun -npernode $mpi_tasks_per_node -np $np $application $options"


###############################################################
### You should not have to change anything below this line ####
###############################################################

cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
        #! Create a machine file:
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\nNodes allocated:\n================"
        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

scancel $SLURM_ARRAY_JOB_ID
