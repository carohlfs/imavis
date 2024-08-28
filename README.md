# Problem-dependent Attention and Effort in Neural Networks: Code and Analysis

This repository contains the code and analysis used for the paper "Problem-dependent attention and effort in neural networks with applications to image resolution and model selection" ([article link](https://www.sciencedirect.com/science/article/abs/pii/S0262885623000707) [arXiv link](https://arxiv.org/abs/2201.01415)). The repository is structured into six sections, each corresponding to different aspects of the analysis and figures presented in the paper.

## Table of Contents
- [1_probabilities](#1_probabilities)
- [2_sample_images](#2_sample_images)
- [3_cost](#3_cost)
- [4_clean](#4_clean)
- [5_correlations](#5_correlations)
- [6_ensemble](#6_ensemble)
- [Installation](#installation)
- [Usage](#usage)

### 1_probabilities
This folder contains Python scripts used to train various models as required and output the full set of model probabilities, chosen classifications, and actual values into CSV files. These outputs are crucial for evaluating the performance and reliability of the models under different conditions.

### 2_sample_images
The scripts in this folder, written in Python and R, are used to select sample images and generate Figures 2 and 3 of the paper. Figure 2 illustrates what sample images look like at different resolution levels, and Figure 3 shows example cases that are easy and hard for the models to classify.

### 3_cost
Python scripts in this folder calculate the data and computation costs associated with different models and images. This data informs the sequential classifiers used in the analysis and is summarized in Table 2 of the paper.

### 4_clean
R scripts that clean and organize data and producing outputs for Table 3 and Figures 4 & 5. Table 3 shows the rates of accuracy of different models, Figure 4 presents histograms of the distributions of model confidence levels across different test cases, and Figure 5 illustrates how model accuracy varies with these confidence levels.

### 5_correlations
R scripts that generate Figure 6, showing correlations between model confidence and accuracy.

### 6_ensemble
This folder contains R scripts that implement two threshold-based estimators. One estimator selects image resolution to reduce data usage (Table 4 & Figure 7), while the other selects models to reduce computation (Table 5 & Figure 8). Both the tables and the figures illustrate the tradeoffs between accuracy and cost.

## Installation
To run the code in this repository, you'll need to install some dependencies as illustrated below.

1. **Install Required Packages**:

- At a Unix command prompt, install the necessary Python packages:
```bash
pip3 install -r requirements.txt
```
- And in an R session's command prompt, install additional packages:
```R
install.packages(c("data.table", "tidyr", "scales", "lmtest", "aod", "parallel", "zoo", "svglite", "rlist"))
```

2. **Set Up ImageNet-V2 Dataset**:
   - This project requires the ImageNet-V2 dataset, which must be downloaded separately due to licensing restrictions. Follow these steps:
     - Visit the [ImageNet download page](https://image-net.org/download.php) and create an account or log in.
     - Request access to the ImageNet-V2 1k dataset. Approval may take some time.
     - After gaining access, ensure you have at least 150GB of free space and download the dataset and extract it to a directory on your local machine:
       ```bash
       mkdir -p ~/datasets/imagenetv2
       tar -xzf path_to_downloaded_file.tar.gz -C ~/datasets/imagenetv2
       ```
     - Update your project configuration to point to the dataset directory:
       ```python
       dataset_path = '~/datasets/imagenetv2'
       ```

**Note**: The procedures for accessing and downloading the dataset may change over time. If you encounter issues, please refer to the latest instructions on the [ImageNet website](https://image-net.org/download.php).


## Usage
To run the model training and probability output script in the `1_probabilities` folder:

```bash
python3 train_models.py
```

## Reference
If you use this code, please cite my paper:

```bibtex
@ARTICLE {Rohlfs2023ProblemDependent,
  AUTHOR = "C. Rohlfs",
  YEAR = "2023",
  TITLE = "Problem-dependent attention and effort in neural networks with applications to image resolution and model selection",
  JOURNAL = "Image and Vision Computing",
  VOLUME = "135",
  PAGES = "104696" }
```

## License
This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
