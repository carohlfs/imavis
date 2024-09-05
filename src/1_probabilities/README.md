# 1_probabilities

This folder contains four subfolders, each of which contains code that extracts data and generates model-forecasted probabilities for a different class of models. The scripts inside these folders are run in sequence by the bash shell script probabilities.sh. That script should be called from a Unix prompt with "run" as the working directory.

Because of access restrictions, the ImageNet data should be downloaded separately. Three files should be saved in tmp/imagenet. Those are:

- ILSVRC2012_img_train.tar,
- ILSVRC2012_img_val.tar, and
- ILSVRC2017_devkit.tar.gz

The remaining datasets download automatically through PyTorch.

The scripts contained inside src/1_probabilities download and extract source data, train models, and produce predictions--labels and probabilities from validation and test cases with different levels of downsampling--which are saved to the tmp/probs folder. These data and intermediate outputs are used by later scripts to generate the graphs and tables that appear in the paper.

As Table 1 in the paper shows, the following models are run for each of the six datasets used:
- MNIST, KMNIST, and Fashion MNIST: Logistic regression and LeNet-5
- SVHN and CIFAR-10: Logistic regression, LeNet-5, DLA-Simple, and ResNet-18
- ImageNet and ImageNet-V2: AlexNet, Inception v3, GoogLeNet, MobileNet v3 small, MobileNet v3 large, VGG 19-bn, DenseNet-201, ResNet-101, EfficientNet b7, EfficientNet b0, Wide ResNet 101-2, and ResNeXt-101-23x8d

- **dlares:** The first subfolder in src/1_probabilities contains code to run DLA-Simple and ResNet-18 on the SVHN and CIFAR-10 datasets. The main script, dlares.py trains and implements the DLA-Simple and ResNet-18 models on CIFAR-10 and SVHN. The remaining scripts in the folder contain additional model-related functions that the main script imports. These scripts modified from a git by kuangliu. The syntax for running this script is as follows, where the possible dataset names include "svhn" and "cifar" and the model names include "dla" and "res":

```bash
python3 ../src/1_probabilities/dlares/dlares.py --dataset "$dataset" --model "$model" --lr "$LEARNING_RATE" $RESUME_FLAG
```

	The intermediate outputs from this code are {$dataset}_{$sample}{$rate}_{$model}preds.csv (the selected class for each case) and {$dataset}_{$sample}{$rate}_{$model}probs.csv (the probabilities for each of the classes). where sample is "train" or "test" and the rates are the image resolution levels of 32, 16, 8, 4, and 2 for the downsampled input. For training, the only rate used is 32.

- **lenet:** This second subfolder in src/1_probabilities contains code to run the LeNet-5 model on each of MNIST, KMNIST, Fashion MNIST, and grayscale versions of SVHN and CIFAR-10. The script lenet.py is modified from a git by erykml. The syntax for running the script is as follows, where the possible dataset names include "mnist" "kmnist" "fashionmnist" "svhn" and "cifar":

```bash
python3 ../src/1_probabilities/lenet/lenet5.py --dataset "$dataset"
````

	The intermediate outputs from this code are {$dataset}_{$sample}{$rate}_preds.csv, {$dataset}_{$sample}{$rate}_probs.csv, and {$dataset}_{$sample}{$rate}_X.csv (the normalized pixel intensities for the different datasets), where sample is train or test and the rates are the image resolution levels of 28, 14, 7, 4, and 2 for mnist, kmnist, and fashionmnist and 32, 16, 8, 4, and 2 for svhn and cifar.

- **logit:** Scripts in this folder extract the raw data from the saved images in ../data, perform principal components analysis to reduce the dimensionality of the pixel intensity values, and then run logistic regression on the principal components. These scripts operate on the original and downsampled files extracted & generated from the previous scripts. The syntax for running is:

```bash
python3 ../src/1_probabilities/logit/savecsv.py
R --vanilla < ../src/1_probabilities/logit/pca.R
R --vanilla < ../src/1_probabilities/logit/logit.R
```

	The outputs saved in the../data folder are ../data/{$dataset}/{$dataset}_{$sample}.csv for each of train and test samples and ../data/{$dataset}.RDS, all for each of the mnist, kmnist, fashionmnist, svhn, and cifar datasets. The intermediate outputs saved in ../tmp/probs include {$dataset}_eigs.RDS (the eigenvectors from the PCAs), {$dataset}_{$sample}{$rate}_pca.RDS (the transformed datasets of principal components), {$dataset}_logits.RDS (the estimated parameters from the logistic regressions), and {$dataset}_{$sample}{$rate}_lprobs.RDS (the forecasted probabilities from those regressions).

- **imagenet:** Scripts in this folder organize some of the ImageNet data and run 12 pre-trained neural network models on the ImageNet and ImageNet-V2 datasets. They require that the training data are already extracted into ../data/imagenet. Some of the relevant extraction steps appear in probabilities.sh. The sort_valid.py script takes the tar file '../tmp/imagenet/ILSVRC2012_img_val.tar' as in input but also requires the validation data's labels, contained in '../tmp/imagenet/ILSVRC2015_clsloc_validation_ground_truth.txt' as well as a file '../tmp/imagenet/map_clsloc.txt' that provides a mapping from 0-999 labels to the class codes. This cleaning script is run as follows:

```bash
python3 ../src/1_probabilities/imagenet/sort_valid.py
```

	Next, we run a script that outputs the actuals (the ground truth classifications) for the training and validation (from ImageNet) and test (from ImageNet-V2) datasets:

```bash
python3 ../src/1_probabilities/imagenet/print_actuals.py
```

	The next and most computationally intensive step is to apply the 12 trained models to produce forecasted values and probabilities on each of the training, validation, and test cases from the ImageNet and ImageNet-V2 data using the imagenet.py script, which is operated as follows, where modelindex values range from 0 to 11, sample can be "train" "val" or "test" and psize is the resolution of the image, which is 256, 128, 64, 32, or 16 (always 256 for training):

```bash
python3 ../src/1_probabilities/imagenet/imagenet.py $modelindex $sample $psize
```

	The outputs saved in the ../tmp/probs folder are ../tmp/probs/{$model}_{$psize}_{$sample}.csv.
