o inet

    - clean.batches.R consolidates the ImageNet & ImageNet-V2 probabilities from the training data & computes accuracy of the different models

    - clean.preds.R consolidates the ImageNet & ImageNet-V2 probabilities from the validation & test data for different image resolutions & saves accuracy rates and cleaned.data.RDS
		
    - prob.graphs.R generates the ImageNet and ImageNet-V2 subfigures plotting sample frequency and accuracy for different ranges of propensity scores, as in Figures 4 & 5 of the paper.
	
o mnist_rgb

    - compare.probs.R generates the corresponding propensity score plots for the MNIST-style and RGB images
