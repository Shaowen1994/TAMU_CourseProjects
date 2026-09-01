#### submit terra jobs for msi calculation

if [ ! -d "Terra_jobs" ]
then
    mkdir Terra_jobs
fi

if [ ! -d "Outputs" ]
then
    mkdir Outputs
fi

### create_job

### create terra job for msi calculation

version=$1
sample=$2

title="${version}_${sample}"

script_input="python train_10FoldCV_gvae.py --m ${version} --r ${sample}"

echo "Script:" ${script_input}
echo "Job Title:" ${title}

echo "#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=${title}
#SBATCH --time=02:00:00              
#SBATCH --ntasks=28
#SBATCH --mem=50G                  
#SBATCH --output=Outputs/output_traning_${title}
#SBATCH --gres=gpu:1                #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --account=122821646257       #Set billing account to 

#First Executable Line
module load Anaconda/3-5.0.0.1
source activate GVAE
${script_input}
source deactivate" > Terra_jobs/Terra_traning_${title}.job

### submit

sbatch Terra_jobs/Terra_traning_${title}.job

