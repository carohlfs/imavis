import numpy as np
import pandas as pd

# Function to load and process datasets
def process_dataset(filenames, dataname, model_name, num_classes=10):
    dfs = [pd.read_csv(filename) for filename in filenames]
    df = pd.concat(dfs, ignore_index=True)
    n_rows = [len(df_) for df_ in dfs]
    
    df['obs'] = np.concatenate([np.arange(1, n + 1) for n in n_rows])
    df['dataset'] = np.repeat(dataname, n_rows)
    df['model'] = model_name
    
    # Find the maximum probability and the corresponding class
    df['phat'] = df.iloc[:, 1:num_classes + 1].max(axis=1)
    df['selection'] = df.iloc[:, 1:num_classes + 1].idxmax(axis=1).apply(lambda x: int(x.replace('V', '')) - 1)
    
    return df[['obs', 'dataset', 'model', 'phat', 'selection']]

# Function to merge datasets with actual labels and determine best/worst
def merge_with_actuals(predictions, actuals_filenames, num_classes=10):
    actuals = pd.concat([pd.read_csv(f) for f in actuals_filenames], ignore_index=True)
    predictions = pd.concat([predictions, actuals.rename(columns={'V1': 'actual'})], axis=1)
    
    # Sort by confidence score
    predictions = predictions.sort_values(by=['dataset', 'phat'], ascending=False)
    
    # Split by correct/wrong predictions
    correct_preds = predictions[predictions['actual'] == predictions['selection']]
    return correct_preds, predictions

# Function to compute best/worst examples
def get_best_worst(splits_correct, splits, n_best=2, n_worst=2):
    worst = pd.concat([df.head(n_worst) for df in splits.values()])
    best = pd.concat([df.tail(n_best) for df in splits_correct.values()])
    return worst, best

# Process LeNet datasets
dataname_mnist = ['mnist', 'kmnist', 'fashionmnist']
rate = 7
filenames_mnist = [f"/path/to/{name}{rate}_test_probs.csv" for name in dataname_mnist]
lenet_df = process_dataset(filenames_mnist, dataname_mnist, 'lenet')

actuals_filenames_mnist = [f"/path/to/{name}_test_actuals.csv" for name in dataname_mnist]
lenet_correct, lenet_splits = merge_with_actuals(lenet_df, actuals_filenames_mnist)

worst_lenet, best_lenet = get_best_worst(lenet_correct, lenet_splits)

# Process ResNet datasets
dataname_resnet = ['svhn', 'cifar']
rate_resnet = 8
filenames_resnet = [f"/path/to/{name}{rate_resnet}_test_resprobs.csv" for name in dataname_resnet]
resnet_df = process_dataset(filenames_resnet, dataname_resnet, 'resnet')

actuals_filenames_resnet = [f"/path/to/{name}_test_actuals.csv" for name in dataname_resnet]
resnet_correct, resnet_splits = merge_with_actuals(resnet_df, actuals_filenames_resnet)

worst_resnet, best_resnet = get_best_worst(resnet_correct, resnet_splits)

# Process DLA datasets
filenames_dla = [f"/path/to/{name}{rate_resnet}_test_dlaprobs.csv" for name in dataname_resnet]
dla_df = process_dataset(filenames_dla, dataname_resnet, 'dla')

actuals_filenames_dla = [f"/path/to/{name}_test_actuals.csv" for name in dataname_resnet]
dla_correct, dla_splits = merge_with_actuals(dla_df, actuals_filenames_dla)

worst_dla, best_dla = get_best_worst(dla_correct, dla_splits)

# Combine ResNet and DLA for comparison
combined_df = pd.concat([resnet_df, dla_df], axis=1)
combined_df['phat_combined'] = 0.5 * (combined_df['phat_resnet'] + combined_df['phat_dla'])
combined_df = combined_df.drop(columns=['phat_resnet', 'phat_dla'])

combined_correct, combined_splits = merge_with_actuals(combined_df, actuals_filenames_resnet)
worst_combined, best_combined = get_best_worst(combined_correct, combined_splits)

# Output best and worst images
print("Best images:")
print(best_combined)
print("\nWorst images:")
print(worst_combined)
