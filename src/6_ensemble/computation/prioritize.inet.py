import pandas as pd
import numpy as np
import pyreadr
from dask import dataframe as dd
from scipy.stats import cumfreq
from pandas import melt

# Read the cleaned data RDS file
result = pyreadr.read_r('cleaned.data.RDS')
cleaned_data = result[None]  # The data will be in the None key

# Filter to size 256 and drop the 'size' column
cleaned_data = cleaned_data[cleaned_data['size'] == 256].drop(columns=['size'])

# Convert 'sample' to a categorical variable with specified levels
cleaned_data['sample'] = pd.Categorical(cleaned_data['sample'], categories=['valid', 'test'], ordered=True)

# Manually setting GMACs and msec values by model
gmacs_values = {
    "resnext101_32x8d": 16.51, "wide_resnet101_2": 22.82, "efficientnet_b0": 0.40167,
    "efficientnet_b7": 5.27, "resnet101": 7.85, "densenet201": 4.37,
    "vgg19_bn": 19.7, "mobilenet_v3_large": 0.22836, "mobilenet_v3_small": 0.06013,
    "googlenet": 1.51, "inception_v3": 2.85, "alexnet": 0.71556
}
msecs_values = {
    "resnext101_32x8d": 67.4, "wide_resnet101_2": 71.2, "efficientnet_b0": 25.5,
    "efficientnet_b7": 126.0, "resnet101": 29.6, "densenet201": 35.6,
    "vgg19_bn": 156.9, "mobilenet_v3_large": 15.0, "mobilenet_v3_small": 10.7,
    "googlenet": 14.5, "inception_v3": 20.8, "alexnet": 6.8
}

# Apply GMACs and msecs values to the data
cleaned_data['gmacs'] = cleaned_data['model'].map(gmacs_values)
cleaned_data['msecs'] = cleaned_data['model'].map(msecs_values)

# Calculate means for the 'nothresh' output
nothresh = cleaned_data.groupby(['sample', 'model']).agg(
    correct=('correct', 'mean'),
    flops=('gmacs', 'mean'),
    msecs=('msecs', 'mean')
).reset_index()

# Save the 'nothresh' output
nothresh.to_pickle("nothresh.pkl")

# Function to prioritize by threshold
def prioritize_thresh(threshold, model_set, cost_var="gmacs"):
    last_model = model_set[-1]
    priority_data = cleaned_data[cleaned_data['model'].isin(model_set)]
    
    # Calculate cumulative cost, flops, msecs, and maximum probabilities
    priority_data['cost'] = priority_data[cost_var]
    priority_data = priority_data.sort_values(by=['sample', 'obs', 'cost'])
    priority_data['cum_cost'] = priority_data.groupby(['sample', 'obs'])['cost'].cumsum()
    priority_data['cum_flops'] = priority_data.groupby(['sample', 'obs'])['gmacs'].cumsum()
    priority_data['cum_msecs'] = priority_data.groupby(['sample', 'obs'])['msecs'].cumsum()
    priority_data['cummax_prob'] = priority_data.groupby(['sample', 'obs'])['prob'].cummax()
    
    # Update cumulative predictions and correct calculations
    priority_data['cummax_pred'] = priority_data['pred']
    priority_data.loc[priority_data['prob'] != priority_data['cummax_prob'], 'cummax_pred'] = np.nan
    priority_data['cummax_pred'] = priority_data.groupby(['sample', 'obs'])['cummax_pred'].ffill()
    priority_data['cummax_correct'] = (priority_data['cummax_pred'].round() == priority_data['actual'].round()).astype(int)
    
    # Filter based on threshold or last model
    priority_data = priority_data[(priority_data['prob'] >= threshold) | (priority_data['model'] == last_model)]
    priority_data['min_cost'] = priority_data.groupby(['sample', 'obs'])['cum_cost'].transform('min')
    priority_data = priority_data[priority_data['min_cost'] == priority_data['cum_cost']]
    
    # Calculate priority means
    priority_means = priority_data.groupby('sample').agg(
        accuracy=('cummax_correct', 'mean'),
        flops=('cum_flops', 'mean'),
        msecs=('cum_msecs', 'mean')
    ).reset_index()
    
    return priority_means

# Function to prioritize across multiple thresholds
def prioritize(model_set, cost_var):
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99]
    priority_sets = pd.concat([prioritize_thresh(thresh, model_set, cost_var) for thresh in thresholds])
    priority_sets['threshold'] = np.repeat(thresholds, 2)  # Each threshold appears for both samples
    return priority_sets

# Full prioritization function
def prioritize_full():
    all_models = [
        "mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "alexnet", "googlenet",
        "inception_v3", "densenet201", "efficientnet_b7", "resnet101", "resnext101_32x8d",
        "vgg19_bn", "wide_resnet101_2"
    ]
    select1 = ["mobilenet_v3_small", "efficientnet_b0", "resnext101_32x8d"]
    select2 = ["mobilenet_v3_large", "efficientnet_b0", "densenet201", "resnet101", "resnext101_32x8d", "wide_resnet101_2"]
    select3 = ["mobilenet_v3_small", "efficientnet_b0", "resnet101", "resnext101_32x8d"]
    select4 = ["mobilenet_v3_large", "efficientnet_b0", "densenet201", "resnext101_32x8d"]

    candidates = ["all1", "all2", "select1", "select2", "select3", "select4"]
    
    # Run prioritize for each model set
    priorities_all1 = prioritize(all_models, cost_var="gmacs")
    priorities_all2 = prioritize(all_models, cost_var="msecs")
    priorities_select1 = prioritize(select1, cost_var="msecs")
    priorities_select2 = prioritize(select2, cost_var="gmacs")
    priorities_select3 = prioritize(select3, cost_var="msecs")
    priorities_select4 = prioritize(select4, cost_var="gmacs")
    
    # Combine results
    priorities = pd.concat([
        priorities_all1, priorities_all2, priorities_select1, priorities_select2,
        priorities_select3, priorities_select4
    ])
    
    priorities['candidates'] = np.repeat(candidates, [len(priorities_all1), len(priorities_all2), len(priorities_select1),
                                                      len(priorities_select2), len(priorities_select3), len(priorities_select4)])
    
    # Melt and spread priorities
    priorities = pd.melt(priorities, id_vars=['candidates', 'sample', 'threshold'])
    priorities = priorities.pivot_table(index=['sample', 'candidates', 'variable'], columns='threshold', values='value').reset_index()
    
    return priorities

# Define max model function
def max_model(model_set):
    max_data = cleaned_data[cleaned_data['model'].isin(model_set)]
    max_data['flops'] = max_data.groupby(['sample', 'obs'])['gmacs'].transform('sum')
    max_data['msecs'] = max_data.groupby(['sample', 'obs'])['msecs'].transform('sum')
    max_data['pmax'] = max_data.groupby(['sample', 'obs'])['prob'].transform('max')
    
    # Filter rows where prob equals pmax
    max_data = max_data[max_data['prob'] == max_data['pmax']]
    max_means = max_data.groupby('sample').agg(
        accuracy=('correct', 'mean'),
        flops=('flops', 'mean'),
        msecs=('msecs', 'mean')
    ).reset_index()
    
    return max_means

# Full max calculation
def max_full():
    all_models = [
        "mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "alexnet", "googlenet",
        "inception_v3", "densenet201", "efficientnet_b7", "resnet101", "resnext101_32x8d",
        "vgg19_bn", "wide_resnet101_2"
    ]
    select1 = ["mobilenet_v3_small", "efficientnet_b0", "resnext101_32x8d"]
    select2 = ["mobilenet_v3_large", "efficientnet_b0", "densenet201", "resnet101", "resnext101_32x8d", "wide_resnet101_2"]
    select3 = ["mobilenet_v3_small", "efficientnet_b0", "resnet101", "resnext101_32x8d"]
    select4 = ["mobilenet_v3_large", "efficientnet_b0", "densenet201", "resnext101_32x8d"]

    candidates = ["all1", "all2", "select1", "select2", "select3", "select4"]
    
    # Max model for each selection
    max_all = max_model(all_models)
    max_select1 = max_model(select1)
    max_select2 = max_model(select2)
    max_select3 = max_model(select3)
    max_select4 = max_model(select4)

    maxes = pd.concat([max_all, max_select1, max_select2, max_select3, max_select4])
    maxes['candidates'] = np.repeat(candidates, [len(max_all), len(max_select1), len(max_select2), len(max_select3), len(max_select4)])
    
    return maxes

# Define the benchmark function
def make_benchmark(model_set):
    benchmark = cleaned_data[cleaned_data['model'].isin(model_set)].groupby(['sample', 'model']).agg(
        accuracy=('correct', 'mean'),
        flops=('gmacs', 'mean'),
        msecs=('msecs', 'mean')
    ).reset_index()

    # Get max accuracy by sample
    benchmark['max_accuracy'] = benchmark.groupby('sample')['accuracy'].transform('max')
    benchmark = benchmark[benchmark['accuracy'] == benchmark['max_accuracy']]
    
    return benchmark[['sample', 'accuracy', 'flops', 'msecs']]

# Full benchmark function
def make_benchmarks():
    all_models = [
        "mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "alexnet", "googlenet",
        "inception_v3", "densenet201", "efficientnet_b7", "resnet101", "resnext101_32x8d",
        "vgg19_bn", "wide_resnet101_2"
    ]
    select1 = ["mobilenet_v3_small", "efficientnet_b0", "resnext101_32x8d"]
    select2 = ["mobilenet_v3_large", "efficientnet_b0", "densenet201", "resnet101", "resnext101_32x8d", "wide_resnet101_2"]
    select3 = ["mobilenet_v3_small", "efficientnet_b0", "resnet101", "resnext101_32x8d"]
    select4 = ["mobilenet_v3_large", "efficientnet_b0", "densenet201", "resnext101_32x8d"]

    candidates = ["all1", "all2", "select1", "select2", "select3", "select4"]

    # Benchmarks for each model set
    benchmark_all = make_benchmark(all_models)
    benchmark_select1 = make_benchmark(select1)
    benchmark_select2 = make_benchmark(select2)
    benchmark_select3 = make_benchmark(select3)
    benchmark_select4 = make_benchmark(select4)

    benchmarks = pd.concat([benchmark_all, benchmark_select1, benchmark_select2, benchmark_select3, benchmark_select4])
    benchmarks['candidates'] = np.repeat(candidates, [len(benchmark_all), len(benchmark_select1), len(benchmark_select2), len(benchmark_select3), len(benchmark_select4)])
    
    return benchmarks

# Generate benchmarks, priorities, and maxes
benchmarks = make_benchmarks()
priorities = prioritize_full()
maxes = max_full()

# Melt maxes for merging
maxes = pd.melt(maxes, id_vars=['sample', 'candidates'])

# Combine priorities with benchmarks and maxes
priorities_combined = pd.concat([
    benchmarks.set_index(['sample', 'candidates']),
    maxes[['sample', 'candidates', 'value']].rename(columns={'value': 'pmax'}).set_index(['sample', 'candidates']),
    priorities.set_index(['sample', 'candidates'])
], axis=1)

# Save the final priorities table
priorities_combined.to_csv("priorities.inet.txt", sep='\t')

# Optional: write filtered priorities (commented out)
# filtered_priorities = priorities_combined[priorities_combined.index.get_level_values('candidates').isin(["all1", "select2", "select4"])]
# filtered_priorities.to_csv("priorities_filtered.inet.txt", sep='\t')