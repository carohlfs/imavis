# imavis
Code used to perform the analyses in "Problem-dependent attention and effort in neural networks with applications to image resolution and model selection." https://arxiv.org/abs/2201.01415

1_probabilities - Python code to train models (as necessary) and to output the full set of model probabilities, chosen classifications, and actuals into csvs.

2_sample_images - Python & R code to pick out the sample images & print Figures 2 & 3.

3_cost - Python code to calculate the data & computation cost associated with different models & images. This information is used in the sequential classifiers and also in Table 2.

4_clean - R code to organize the data and produce outputs for Table 3 and Figures 4 & 5.

5_correlations - R code to produce Figure 6

6_ensemble - R code to implement the two threshold-based estimators, one that selects image resolution to reduce data usage (Table 4 & Figure 7) and another that selects models to reduce computation (Table 5 & Figure 8).
