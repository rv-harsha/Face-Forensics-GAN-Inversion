#!/bin/bash 

# progress bar function
prog() {
    local w=80 p=$1;  shift
    # create a string of spaces, then change them to dots
    printf -v dots "%*s" "$(( $p*$w/100 ))" ""; dots=${dots// /.};
    # print those dots on a fixed-width space plus the percentage etc. 
    printf "\r\e[K|%-*s| %3d %% %s" "$w" "$dots" "$p" "$*"; 
}

dataset_path=/nas/vista-hdd01/users/raidurga/workspace/aifi/dataset/celebA/images
batch_path=/nas/vista-hdd01/users/raidurga/workspace/aifi/dataset/celebA/batch-info

batch_size=5
file_counter=0
batch_counter=1
batch_stop=400 #6000

if [ -z "$1" ]  # True if the length of string is zero.
    then
        cmd=`ls $dataset_path | shuf`
else
    files=()
    while read p; do
        files+=($p)
    done <$1
    cmd="${files[@]}"
fi

for filename in ${cmd}
do  
    batch=$(($batch_counter*100/$batch_stop))
    prog "$batch" Creating batch files. Please wait...
    batch_csv=$batch_path/batch-$batch_counter.csv
    echo "$filename" >> "$batch_csv"
    (( file_counter += 1 ))
    if [ "$file_counter" -eq "$batch_size" ]; then
        (( batch_counter += 1 ))
        file_counter=0
    fi
    if [ "$batch_counter" -eq "$batch_stop" ]; then
        break
    fi
done