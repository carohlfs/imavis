import pandas as pd
import numpy as np

def clean_preds(model_name):
    sizes = [16, 32, 64, 128, 256]
    model_files = [f"{model_name}.{size}.csv" for size in sizes]
    
    # Read CSV files and concatenate into a single DataFrame
    model_data = pd.concat([pd.read_csv(file, header=None) for file in model_files], ignore_index=True)
    
    # Assigning column names and creating 'variable' and 'id' columns
    model_data['variable'] = 'prob'
    model_data['id'] = model_data.index + 1
    
    # Modify 'variable' based on even/odd 'id'
    model_data.loc[model_data['id'] % 2 == 0, 'variable'] = 'pred'
    model_data['id'] = np.floor((model_data['id'] + 1) / 2).astype(int)
    
    # Spread/reshape to wide format
    model_data = model_data.pivot(index='id', columns='variable', values=0).reset_index()
    
    # Add 'sample', 'obs', 'size', and 'model' columns
    model_data['sample'] = np.repeat(np.concatenate([['valid'] * 50000, ['test'] * 10000]), 5)
    model_data['obs'] = np.tile(np.concatenate([np.arange(1, 50001), np.arange(1, 10001)]), 5)
    model_data['size'] = np.repeat(sizes, 60000)
    model_data['model'] = model_name
    
    return model_data

# Model names as in the original R script
model_names = ["resnext101_32x8d", "wide_resnet101_2", "efficientnet_b0", "efficientnet_b7",
               "resnet101", "densenet201", "vgg19_bn", "mobilenet_v3_large",
               "mobilenet_v3_small", "googlenet", "inception_v3", "alexnet"]

# Clean the data for each model and concatenate them
cleaned_data = pd.concat([clean_preds(model_name) for model_name in model_names], ignore_index=True)

# Read the actuals data
valid_actuals = pd.read_csv('valid_actuals.csv', header=None, names=['actual'])
test_actuals = pd.read_csv('test_actuals.csv', header=None, names=['actual'])

# Add 'sample' and 'obs' columns to actuals
valid_actuals['sample'] = 'valid'
valid_actuals['obs'] = np.arange(1, 50001)
test_actuals['sample'] = 'test'
test_actuals['obs'] = np.arange(1, 10001)

# Concatenate the actuals data
actuals = pd.concat([valid_actuals, test_actuals], ignore_index=True)

# Merge actuals with cleaned_data
cleaned_data = pd.merge(cleaned_data, actuals, on=['sample', 'obs'])

# Add 'correct' column based on whether the prediction matches the actual value
cleaned_data['correct'] = (cleaned_data['actual'] == cleaned_data['pred']).astype(int)

# Save the cleaned data to an RDS equivalent (Pickle in Python)
cleaned_data.to_pickle('cleaned_data.pkl')

# Calculate the mean correctness rates grouped by 'sample', 'size', and 'model'
rates = cleaned_data.groupby(['sample', 'size', 'model']).agg({'correct': 'mean'}).reset_index()

# Pivot the rates DataFrame to wide format, similar to spread in R
rates_pivot = rates.pivot(index=['sample', 'model'], columns='size', values='correct').reset_index()

# Reorder columns as per the original R script
rates_pivot = rates_pivot[['sample', 'model', 256, 128, 64, 32, 16]]

# Save the rates DataFrame to a text file
rates_pivot.to_csv('rates.txt', sep='\t', index=False)