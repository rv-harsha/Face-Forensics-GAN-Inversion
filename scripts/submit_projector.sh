#!/bin/bash 

IMAGES="/nas/vista-hdd01/users/raidurga/workspace/styleGAN2/dataset/Temp/*"

out_path=/nas/vista-hdd01/users/raidurga/outputs/
mkdir -p $out_path

run_id=002
exp_id=rnw_run_${run_id}      
exp_name=ai2ai_${exp_id}
ncpu=4

qos=limited
part=ALL
# qos=premium_memory
# part=large_gpu
mem=16G
account=other

expts_path=${out_path}${exp_name}/expts/
logs_path=${out_path}${exp_name}/logs/
sbatch_path=${out_path}${exp_name}/sbatch/

mkdir -p $expts_path
mkdir -p $logs_path
mkdir -p $sbatch_path
counter=0

reg_loss_wgts=(1e1 1e2 1e3 1e4 1e5 1e6 1e7 1e8 1e9)

for file in $IMAGES; do
    fileName="$(basename -s .png  "$file")"
    path=$file

    for rnw in "${reg_loss_wgts[@]}" ; do
        
        job_path=${expts_path}${fileName}/${rnw}/
        mkdir -p "$job_path"
        
        job_name=rnw_${fileName}_${rnw}
        log_file=${logs_path}${job_name}.log
        sbatch_file=${sbatch_path}${fileName}.sbatch
        
        cmd="sbatch --account $account --qos $qos --partition $part --requeue --time=1-23:59:59 --gres=gpu:1 --job-name=$exp_name!$job_name --output=$log_file --gpus=1 --spread-job  --gpus-per-node=1 --mem-per-gpu=8G  --cpus-per-task=$ncpu --export=jid=$job_name,job_path=$job_path,target=$path,rnw=$rnw run_projector.sh 2>&1 | tee -a $sbatch_file"

        echo "$cmd" >> "$sbatch_file"
        echo "$job_name"
        eval "$cmd"
        (( counter += 1 ))
        echo "Submitted $counter jobs. Sleeping before submitting the next one or exiting!"
        sleep 5s
    done
done