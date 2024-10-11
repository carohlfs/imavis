import pandas as pd
import numpy as np
import pyreadr
from dask import dataframe as dd
from dask import compute
from scipy.stats import cumfreq
from pandas import melt

# Function to prioritize thresholds
def prioritize_thresh(threshold, datasets, model_set, adj=False):
    last_model = model_set[-1]
    priority_data = small_results.query("dataset in @datasets and model in @model_set and sample == 'test'")
    
    if adj:
        priority_data['phat'] = priority_data['phat.new']
    
    priority_data = priority_data.sort_values(by=['dataset', 'id', 'gmacs'])
    
    # Cumulative sum columns
    priority_data['cum.flops'] = priority_data.groupby(['dataset', 'id'])['gmacs'].cumsum()
    priority_data['cum.msec'] = priority_data.groupby(['dataset', 'id'])['msec'].cumsum()
    priority_data['cummax.phat'] = priority_data.groupby(['dataset', 'id'])['phat'].cummax()
    
    priority_data['cummax.yhat'] = priority_data['yhat']
    priority_data.loc[priority_data['phat'] != priority_data['cummax.phat'], 'cummax.yhat'] = np.nan
    priority_data['cummax.yhat'] = priority_data.groupby(['dataset', 'id'])['cummax.yhat'].ffill()
    
    priority_data['cummax.correct'] = (priority_data['cummax.yhat'].round() == priority_data['y'].round()).astype(int)
    
    priority_data = priority_data.query("phat >= @threshold or model == @last_model")
    priority_data['minflops'] = priority_data.groupby(['dataset', 'id'])['cum.flops'].transform('min')
    
    priority_data = priority_data.query("minflops == cum.flops")[['dataset', 'id', 'model', 'cummax.phat', 'cummax.correct', 'cum.msec', 'cum.flops']]
    
    priority_means = priority_data.groupby('dataset').agg(
        accuracy=('cummax.correct', 'mean'),
        flops=('cum.flops', 'mean'),
        msec=('cum.msec', 'mean')
    ).reset_index()
    
    return priority_means

# Function to handle multiple thresholds
def prioritize(datasets, model_set, adj=False):
    thresholds = [0.7, 0.8, 0.9, 0.95, 0.975, 0.99]
    priority_sets = [prioritize_thresh(thresh, datasets, model_set, adj) for thresh in thresholds]
    
    thresh_vec = np.repeat(thresholds, [len(df) for df in priority_sets])
    priority_sets = pd.concat(priority_sets)
    priority_sets['threshold'] = thresh_vec
    priority_sets = pd.melt(priority_sets, id_vars=['dataset', 'threshold'])
    
    priority_sets['phat'] = 'adj' if adj else 'unadj'
    priority_sets = priority_sets.pivot_table(index=['dataset', 'variable'], columns='threshold', values='value')
    
    return priority_sets.reset_index()

# Full prioritization process
def prioritize_full():
    mnist_datasets = ["mnist", "kmnist", "fashionmnist"]
    mnist_models = ["logit", "lenet"]

    rgb_datasets = ["svhn", "cifar"]
    rgb_full = ["logit", "lenet", "resnet", "dla"]
    rgb_neural = ["resnet", "dla"]

    # Prioritizations
    p_mnist = prioritize(mnist_datasets, mnist_models)
    p_mnist_adj = prioritize(mnist_datasets, mnist_models, adj=True)
    p_rgb_full = prioritize(rgb_datasets, rgb_full)
    p_rgb_full_adj = prioritize(rgb_datasets, rgb_full, adj=True)
    p_rgb_neural = prioritize(rgb_datasets, rgb_neural)
    p_rgb_neural_adj = prioritize(rgb_datasets, rgb_neural, adj=True)

    priority = pd.concat([p_mnist, p_mnist_adj, p_rgb_full, p_rgb_full_adj, p_rgb_neural, p_rgb_neural_adj], ignore_index=True)
    model_specs = ["full", "full", "full", "full", "neural", "neural"]
    priority['candidates'] = np.repeat(model_specs, [len(p_mnist), len(p_mnist_adj), len(p_rgb_full), len(p_rgb_full_adj), len(p_rgb_neural), len(p_rgb_neural_adj)])

    priority['phat'] = pd.Categorical(priority['phat'], categories=["unadj", "adj"], ordered=True)
    priority = priority[['dataset', 'candidates', 'phat', 'variable', '0.7', '0.8', '0.9', '0.95', '0.975', '0.99']]
    
    return priority

# Reading RDS file
result = pyreadr.read_r('small.results.RDS')  # Use pyreadr to read RDS files
small_results = result[None]  # The object returned by pyreadr is a dictionary

small_results = small_results[small_results['sample'] == 'test']

# Replacing model-specific values
small_results.loc[small_results['model'] == 'logit', 'msec'] = 0.0
small_results.loc[small_results['model'] == 'lenet', 'msec'] = 0.1
small_results.loc[small_results['model'] == 'dla', 'msec'] = 5.8
small_results.loc[small_results['model'] == 'resnet', 'msec'] = 4.8

# Performing prioritization, max calculations, and benchmarks
priorities = prioritize_full()
# Similarly implement `max.full()` and `make.benchmarks()` functions if needed

# Save the final priorities as a text file
priorities.to_csv("priorities.small.txt", sep='\t', index=False)