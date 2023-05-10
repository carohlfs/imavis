- logit - runs logistic regression model on MNIST, Fashion-MNIST, KMNIST, SVHN, & CIFAR-10
- lenet - runs LeNet-v5 model on MNIST, Fashion-MNIST, KMNIST, SVHN, & CIFAR-10
- dlares - runs DLA-Simple & ResNet-18 models on SVHN & CIFAR-10
- inet - runs 12 NN models on ImageNet & ImageNet-V2

The logit folder contains R scripts for cleaning the data & performing logistic regression.

- logit

	o downsample.R
		- consumes training & test CSVs - raw data with pixel intensities & classifications
		- saves RDS files for main data (mnist.RDS through cifar.RDS)
		- saves <dataname><downsamplerate>.RDS files for all combinations of dataname & downsample rate.

	o pca.R
		- generates principal components for actual & downsampled pixel intensity data.
		- full-sized data are always used in training, so just one set of eigenvectors is produced for each dataset
		- the eigenvectors are used to produce datasets of the PCAs for the actual and downsampled datasets; these are saved as <dataname>.ds.pcas.RDS files for each dataset.
		- the non-downsampled training data PCAs are also saved in a single file for all datasets

	o logit.R
		- This script generates formulas and runs regressions (linear & logistic) for each dataset and number of principal components, and it produces forecasted probabilities for each.
		- The lm & glm objects are saved in <data>_regs<factors>.RDS and <data>_logits<factors>.RDS. Each one contains a list of ten class-specific objects.
		- The forecasted probabilities are saved as <data><ds_rate>_<sample>_probs<factors>.RDS for the linear regressions and <data><ds_rate>_<sample>_lprobs<factors>.RDS for the logistic.
		- Only the logistic regressions on the full-sized training images are used. The numbers of factors are 100 for the MNIST-style images and 300 for the RGB images.

The lenet and dlares subfolders here include modified versions of the original PyTorch code from other researchers' publicly available implementations. In both cases, the scripts were modified slightly so that the probabilities from the models are output as csvs.

- lenet

	o implementation taken from https://github.com/erykml/medium_articles/blob/master/Computer%20Vision/lenet5_pytorch.ipynb
	o The five files here are variations on the Jupyter notebook shown there.
	o In all cases, output includes csvs with actuals, probabilities, & X matrices for train & test sets.

- dlares

	o implementation taken from: https://github.com/kuangliu/pytorch-cifar
	o The four files here are replacements for main.py; the same supporting files should be used
	o cifar uses CIFAR-10 data, svhn uses the SVHN
	o DLA implements DLA-Simple model, RES implements ResNet-18
	o In all cases, output includes csvs with actuals & probabilities for train & test sets

The inet folder includes scripts that are run to pull pre-trained model parameters from PyTorch for different popular models that run on ImageNet. The data are assumed to already be saved in the appropriate location. The code simply generates the model projections (including probabilities) for the training & validation sets as well as the "test" set from Recht et al (2019).

- inet

	o print_actuals.py prints the actuals from training, validation (ImageNet) and test (ImageNet-V2) datasets.
	o usage for train_psize.py is:
		python3 train_psize.py <modelnumber>
		- model number is the index value from the "model_names" list shown early in the script
		- outputs model-specific csvs, one with probabilities and another with start & end times.

	o usage for test_psize.py is:
		python3 test_psize.py <modelnumber> <imagesize>
		- model number is index value from list
		- imagesize is the downsampled size: (256, 128, 64, 32, or 16)
		- outputs model- & size-specific csv with probabilities for both validation & test sets.
