# AIFI

Analysis of Inverted Face Images (AIFI) using GAN inversion by the projection of images onto its latent space. The projected images are matched to their original image using different similarity metrics, face detection models.

## Setup

### Conda Environments

Note: Environment files have already been created, so you can install using them directly (refer creating environments from [Appendix](#####creating-environments)). The environment files have been placed under `envs` directory. You can use the files to create a new environement.

#### PyTorch

* This environment would be required to execute any code with respect to latent space projection of target images using StyleGAN2.
* This environment needs to be acivated to run batch projections directly (ie .. without using shell script `submit_batch_projector.sh `but using `run_projector.py`).
* Create environment with suitable name: <env_name>
* ```
  conda create --name <env_name> anaconda python=3.7 gcc_linux-64=5.4.0 gxx_linux-64=5.4.0 -c anaconda -y
  ```
* Activate conda environment and install project dependencies

  ```
  # Reload conda profile to refresh environment info
  source ~/miniconda3/etc/profile.d/conda.sh

  # Activate conda environment
  conda activate <env_name>

  # Install required conda packages
  conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch -y

  # Install imageio-ffmpeg using conda-forge channel
  conda install imageio-ffmpeg -c conda-forge

  # Install pip dependencies
  pip install pyspng
  ```
* Install C++ compiler libraries

  ```
  cd $CONDA_PREFIX/bin
  ln -s x86_64-conda_cos6-linux-gnu-cpp cpp
  ln -s x86_64-conda_cos6-linux-gnu-gcc gcc
  ln -s x86_64-conda_cos6-linux-gnu-g++ g++
  ln -s x86_64-conda_cos6-linux-gnu-c++ c++
  cd 
  conda deactivate <env_name>
  ```

#### DeepFace

* Create Conda Tensorflow environment but install all packages with pip

  ```
  conda create -n <deepface_env_name> python=3.7.10 dlib -c conda-forge

  # Activate conda environment
  conda activate <deepface_env_name>

  # Here deepface is a library installed using pip only
  pip install deepface
  ```

#### Git (only for base)

* ```
  conda activate base
  conda install git -c conda-forge -y
  git config --global http.sslVerify false
  git clone https://gitlab.vista.isi.edu/raidurga/aifi.git
  ```

## Code Base

### Folder Structure

#### `config`

The default configuration file (yml) used in the project is present under the folder. The default config file can be customized and provided as CLI argument to the project. The main configurations for dataset, projector, noise weights and face mathcing/verification params are present in this file.

#### `dataset`

##### Add a new dataset

* Create a dataset folder with a name : `<dataset_name>`
* Create a `images` and `batch-info` folder under that directory
* Add all the images under `images` directory
* Create a CSV file (<dataset_name>.csv) with a list of all image filenames
* Modify the configuration file as shown below:

  ```
  dataset_params:
    use: <dataset_name>
    root_dir: dataset    # Do not change this value
    <dataset_name>: 
      images_dir: images
      csv_filename: <dataset_name>.csv
  ```

#### `dnnlib`

Copy the `dnnlib` folder from StyleGAN2 repository as it is without any modifications

#### `envs`

The exported environment files for both the enviornments are placed under this directory. They can be used to create new environments with this these files (refer creating environments from [Appendix](#####creating-environments)).

#### `scripts`

* `create_batches.sh`: Provide suitable values to the varaiables as shown below to creates batches of size 5 using the images under the `images` directory of a dataset. The script will create CSV files for each batch and place it under `batch-info.`
  ```
  dataset_path=/nas/vista-hdd01/users/raidurga/workspace/aifi/dataset/celebA/images
  batch_path=/nas/vista-hdd01/users/raidurga/workspace/aifi/dataset/celebA/batch-info

  batch_size=5
  file_counter=0
  batch_counter=1
  batch_stop=400 
  ```

#### `torch_utils`

Copy the `torch_utils` folder from StyleGAN2 repository as it is without any modifications

#### `utils`

* `re-submit-jobs.py`: This script provides a list of batch files which have images that have not been projected and verified successfully. Here the `job-status.csv` consists of individual job status for every batch. A left-outer join is performed to fetch the list of INCOMLETE batch jobs that needs to be reprocessed.

## Execution

### Project and Verify

* `project_and_verify.py :` This is the client code to be consumed by an API or external function. This triggers projection and verification sequentially. It assumes GPU is available with conda environments setup.

  ```
  # Activate the enviornment for projection
  conda activate test

  # Provide the images as parameters to this python script
  python project_and_verify.py 1.jpg 2.jpg 3.jpg 4.jpg 5.jpg

  # A UUID will be generated as shown below. The output will be saved under 
  # dataset/<uuid>/execution. Please check in that directory for results. 
  Please save the below unique id (UUID) generated for this request.
  9169a009-698b-41ab-bebf-b0eb7b8e7d8c

  ```

### Batch Projector

* `submit_batch_projector.sh :` submit the batch files created under batch-info directory to SLURM for job creation and scheduling. Change the values of the variables accordingly.
* `run_batch_projector.sh :` Once the job is created, for every job, this script is triggered. It activates the conda environment to project the images as well as verify the projected images with target image.
* `run_projector.py :` Client function for `projector.py` class. This function is invoked by `run_batch_projector.sh` file for execution of projector for every batch of images.

### Face verification

* `face_verify.py :` To only verify the projections using differnt models for an already run experiment, use this script. Provide the required CLI arguements accordingly. Two functions are defined to verify a batch of images individually or all the batches for a given experiment.

## Monitoring

* Navigate to `/nas/vista-hdd01/users/<username>/outputs/ai2ai_run_005/expts`: This directory has execution outputs ie ... the target image and projected images for different regularization noise weights. The structure is as shown below:
  * `<batch_number>/<target_image_name>/<reg_noise_weights>/projected.jpg`
* To view pending jobs: `squeue -u raidurga | grep PD`
* To view running jobs: `squeue -u raidurga | grep R`
* To cancel pending jobs: `squeue -u <username> | grep PD | awk '{print $1}' | xargs -n 1 scancel`

## Reprocessing

* Generate the `reprocess.csv` using the `re-submit-jobs.py` script under `utils` directory. Modify the paths in the python scripts accordingly.
* Now delete all the CSV files under `batch-info` directory.
* Provide the `reprocess.csv `as a CLI input to `create_batches.sh` under `scripts` diectory. This will create new CSV batch files inder `batch-info`. Check the paths, variable values and set them accordingly.
* Run the submit `submit_batch_projector.sh` file.

## Appendix

### Conda

#### Commands

##### Creating Environments

* Create an environment with a specific version of Python version: `conda create -n <new_env> python=3.6. `The first line of the `yml `file sets the new environment's name.

##### Exporting Environments

* To a yml file: `conda env export | grep -v "^prefix: " > deepface-env.yml`. The first line of the `yml `file sets the new environment's name.
* Without build numbers: `conda env export --no-builds | grep -v "^prefix:" > deepface-env.yml`
* To a .txt file: `conda list --explicit > deepace-env.txt`

##### Restore Evironments

* From .txt file: `conda create -name <env_name> -f deepface-env.txt`
* From .yml file: `conda env --name <env_name> create -f environment.yml`

### Git

#### Installation

Needs to be done one time only.

```
conda activate base
conda install git -c conda-forge -y
git config --global http.sslVerify false
```

#### Remote Configuration

```
# List fetch and push urls
git remote -v

# Rename local repo origin to aifi
git remote rename origin aifi
```

#### Clone

`git clone https://gitlab.vista.isi.edu/raidurga/aifi.git`

#### Commit and Push

```
# Refer https://git-scm.com/book/en/v2/Git-Basics-Working-with-Remotes
git add .
git commit -m "Added python and shell source files"

# Optinal configurations to recieve updates
git config --global user.name "<Your Name>" user.email <email_id>

# Push local changes to remote
git push aifi master
```
