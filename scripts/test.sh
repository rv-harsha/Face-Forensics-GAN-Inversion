basename -s .gz /backups/14-nov-2019/backups.tar.gz
#
# Bash get filename from path and store in $my_name variable 
#
name1="$(basename /backups/14-nov-2019/img_001.png)"
name2="$(basename -s .png /backups/14-nov-2019/img_001.png)"
echo "Filename without .png extension : $name1"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

echo $'\n\n----------------------------------------'
echo Setting up projector ...
echo ''
echo "  "Host Name: "               $(hostname)"
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
echo "$(which python)"
conda deactivate