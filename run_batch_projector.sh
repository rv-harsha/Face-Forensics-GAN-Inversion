#!/bin/bash

echo "$SLURM_JOB_ID,${jid},STARTED" >> "${jobfile}"

echo $'\n---------- STARTING JOB EXECUTION ----------\n'
# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh

conda activate test

echo $'\n\n----------------------------------------'
echo Setting up projector ...
echo ''
echo "  "Host Name: "               $(hostname)"
echo "  "Job Id: "                  ${jid}"
echo "  "SLURM Job Id: "            $SLURM_JOB_ID"
echo "  "Job Path: "                ${job_path}"
echo "  "CUDA Visible Devices: "    $CUDA_VISIBLE_DEVICES"
echo "  "TMPDIR 'for' the job: "      $TMPDIR"
echo "  "Conda environment: "       $CONDA_PREFIX"
export TORCH_EXTENSIONS_DIR=$TMPDIR
echo "  "Torch Extensions dir:"     $TORCH_EXTENSIONS_DIR"
echo $'----------------------------------------\n\n'

SOURCE_PATH=$PWD

echo "$SLURM_JOB_ID,${jid},PROJECTING" >> "${jobfile}"

python $SOURCE_PATH/run_projector.py \
    --outdir "$job_path" \
    --dataset_csv "$csv"

echo $'----------------------------------------\n\n'
conda deactivate

echo "$SLURM_JOB_ID,${jid},VERIFYING" >> "${jobfile}"

DF_PYTHON=~/miniconda3/envs/deepface/bin/python
$DF_PYTHON $SOURCE_PATH/face_verify.py \
    --outdir "$job_path" 

echo $'\n\n---------- COMPLETED JOB EXECUTION ----------'

echo "$SLURM_JOB_ID,${jid},COMPLETED" >> "${jobfile}"

retVal=$?
if [ $retVal -ne 0 ]; then
    exit $retVal
fi