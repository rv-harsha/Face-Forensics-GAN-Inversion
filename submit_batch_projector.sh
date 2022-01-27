#!/bin/bash 

# progress bar function
prog() {
    local w=80 p=$1;  shift
    # create a string of spaces, then change them to dots
    printf -v dots "%*s" "$(( $p*$w/100 ))" ""; dots=${dots// /.};
    # print those dots on a fixed-width space plus the percentage etc. 
    printf "\r\e[K|%-*s| %3d %% %s" "$w" "$dots" "$p" "$*"; 
}

batch_info_path=/nas/vista-hdd01/users/raidurga/workspace/aifi/dataset/celebA/batch-info
out_path=/nas/vista-hdd01/users/raidurga/outputs/
mkdir -p $out_path

exp_id=run_005     
exp_name=ai2ai_${exp_id}
ncpu=2

qos=flexible
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
total_jobs=6000
max_running_jobs=20
wait_time=10800s  # 60s * 60 mins * 3 hrs = 10800s

jobfile=${expts_path}job-status.csv

if [ -z "$1" ]  # True if the length of string is zero.
    then
        cmd=`ls $batch_info_path | sort -n`
else
    files=()
    while read p; do
        files+=($p)
    done <$1
    cmd="${files[@]}"
fi

for filename in ${cmd}; do
    inp_fname=(${filename//./ })  
    bch_id=${inp_fname[0]}

    job_path=${expts_path}${bch_id}/
    mkdir -p "$job_path"

    job_name=rnw_${bch_id}
    log_file=${logs_path}${job_name}.log
    sbatch_file=${sbatch_path}${job_name}.sbatch

    cmd="sbatch --account $account --qos $qos --partition $part --requeue --time=1-23:59:59 --gres=gpu:1 --job-name=$exp_name!$job_name --output=$log_file --gpus=1  --gpus-per-node=1 --mem-per-gpu=$mem  --cpus-per-task=$ncpu --export=jid=$job_name,job_path=$job_path,jobfile=$jobfile,csv=$filename run_batch_projector.sh 2>&1 | tee -a $sbatch_file"

    echo "$cmd" >> "$sbatch_file"
    echo "$job_name"
    eval "$cmd"
    (( counter += 1 ))

    # percent_jobs=$(($counter*100/$total_jobs))
    # prog "$percent_jobs" Submitted $counter jobs. Please wait...
    echo "Submitted $counter jobs. Sleeping before submitting the next one or exiting!"

    result=$(echo "$counter%$max_running_jobs" |bc -l)
    # result1=${result/.*}

    if [ $(expr $counter % $max_running_jobs) != "0" ]; then
        sleep 2s
    else
        echo "Sleeping for $wait_time"
        sleep $wait_time
    fi
done