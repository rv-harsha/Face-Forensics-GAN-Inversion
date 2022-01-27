#!/bin/bash
echo $'\n---------- STARTING JOB EXECUTION ----------\n'
# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh

conda activate test

echo $'\n\n----------------------------------------'
echo Setting up projector ...
echo ''
echo "  "Host Name: "               $(hostname)"
echo "  "Job Id: "                  ${jid}"
echo "  "Job Path: "                ${job_path}"
echo "  "CUDA Visible Devices: "    $CUDA_VISIBLE_DEVICES"
echo "  "TMPDIR 'for' the job: "      $TMPDIR"
echo "  "Conda environment: "       $CONDA_PREFIX"
export TORCH_EXTENSIONS_DIR=$TMPDIR
echo "  "Torch Extensions dir:"     $TORCH_EXTENSIONS_DIR"
echo ''
echo Allocated GPU Stats
echo ''
echo "$(nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.free,memory.total --format=csv)"
echo $'----------------------------------------\n\n'

SOURCE_PATH=$PWD
# PYTHON=$(which python)
# target=/nas/vista-hdd01/users/raidurga/workspace/styleGAN2/dataset/Temp/${target}.png
# Run training script
python $SOURCE_PATH/projector.py \
    --outdir "$job_path" \
    --regularize_noise_weight "$rnw" \
    --target "$target" 

echo $'----------------------------------------\n\n'
conda deactivate

# conda activate deepface
DF_PYTHON=~/miniconda3/envs/deepface/bin/python
$DF_PYTHON $SOURCE_PATH/face_verify.py \
    --outdir "$job_path" \
    --reg_noise_wgt "$rnw" 

# conda deactivate

echo $'\n\n---------- COMPLETED JOB EXECUTION ----------'

retVal=$?
if [ $retVal -ne 0 ]; then
    exit $retVal
fi

