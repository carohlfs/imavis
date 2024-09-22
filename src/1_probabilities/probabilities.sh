#!/bin/bash

# ImageNet tar file directory. Files in this directory should include
# ILSVRC2012_img_train.tar, ILSVRC2012_img_val.tar, and ILSVRC2012_devkit.tar.gz
DOWNLOAD_DIR = "../tmp/imagenet"

# Set the learning rate and other common arguments
LEARNING_RATE=0.1
RESUME_FLAG=""

####### DLA and ResNet-18 #######

# Lists of datasets and models for dlares.py
datasets=("svhn" "cifar")
models=("dla" "res")

# Loop over all combinations of datasets and models
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        echo "Running dlares.py with dataset=$dataset and model=$model"
        python3 ../src/1_probabilities/dlares/dlares.py --dataset "$dataset" --model "$model" --lr "$LEARNING_RATE" $RESUME_FLAG
    done
done

####### LeNet-5 #######

# List of datasets for lenet5.py
lenet_datasets=("mnist" "kmnist" "fashionmnist" "svhn" "cifar")

# Loop over all datasets for lenet5.py
for dataset in "${lenet_datasets[@]}"; do
    echo "Running lenet5.py with dataset=$dataset"
    python3 ../src/1_probabilities/lenet/lenet5.py --dataset "$dataset"
done

####### Logistic Regression #######

# Organize image data into csvs that live in ../data folder
python3 savecsv.py
# Reduce dimensionality of pixel matrices with PCA
R --vanilla < ../src/1_probabilities/logit/pca.R
# Run logistic regression using the first 100 or 300 principal components
R --vanilla < ../src/1_probabilities/logit/logit.R

####### Models on ImageNet and ImageNet-V2 Data #######

### Unzip ImageNet Data ###

# The ImageNet data can't be directly downloaded through PyTorch, so we obtain them separately.

## Training ##

# tar file of training images comes from: https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
# After receiving approval from the ImageNet site, download & save in the ../tmp/imagenetv2 folder.

# Create the destination directories for the intermediate and final training data
mkdir -p ../tmp/imagenet/ ../data/imagenet/train

# Extract the main training tar file (which contains class-specific tar files) into the temporary directory
tar -xf "$DOWNLOAD_DIR"/ILSVRC2012_img_train.tar -C ../tmp/imagenet/train

for tar_file in ../data/imagenet/train/*.tar; do
    # Get the class name by stripping the .tar extension
    class_name=$(basename "$tar_file" .tar)
    # Create the class-specific directory
    mkdir -p "../data/imagenet/train/$class_name"
    # Extract the tar file into the class-specific directory
    tar -xf "$tar_file" -C "../data/imagenet/train/$class_name"
    rm "$tar_file"
done

## Validation ##

# The validation data aren't sorted into class-specific tar files, so we need supplemental info.

# Extract the metadata into the ../data/imagenet directory
tar -xzf "$DOWNLOAD_DIR"/ILSVRC2017_devkit.tar.gz ../data/imagenet
# PyTorch also does a check to ensure the tar.gz is in the folder.
cp "$DOWNLOAD_DIR"/ILSVRC2017_devkit.tar.gz ../data/imagenet

# Then run a script that extracts the output files into label-specific folders like with the training data.
python3 ../src/1_probabilities/imagenet/sort_valid.py

### Print Actuals from ImageNet and ImageNet-V2 Data ###

# Save the human labels for the training & validation sets
python3 ../src/1_probabilities/imagenet/print_actuals.py

### Compute Predictions and Probabilities from ImageNet and ImageNet-V2 Data ###

# Now obtain the predictions and probabilities for for the different models.
# We do training plus downsampled versions of the validation and test data.

# List of PSIZE values
PSIZE_values=(256 128 64 32 16)

# Model indices (0-based index for the elements in model_names)
model_indices=$(seq 0 11)  # There are 12 models, so index 0 to 11

# Loop over all combinations of model indices and PSIZE values
for index in $model_indices; do
    echo "Running imagenet.py for training sample with model index=$index"
    python3 ../src/1_probabilities/imagenet/imagenet.py "$index" train
    for PSIZE in "${PSIZE_values[@]}"; do
        echo "Running imagenet.py for validation sample with PSIZE=$PSIZE, model index=$index"
        python3 ../src/1_probabilities/imagenet/imagenet.py "$index" val "$PSIZE"
        echo "Running imagenet.py for test sample with PSIZE=$PSIZE, model index=$index"
        python3 ../src/1_probabilities/imagenet/imagenet.py "$index" test "$PSIZE"
    done
done
