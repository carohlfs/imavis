import numpy as np
import pandas as pd

# Helper function for softmax calculation, returning only the probability of the selected class
def softmax_selected(df, num_classes):
    """
    Calculate softmax for the selected class (maximum logit).
    Args:
    - df: DataFrame containing logits (pre-softmax scores) for each class.
    - num_classes: Number of classes.
    Returns:
    - phat: Softmax probability for the selected class (maximum logit).
    - selection: The index of the selected class.
    """
    # Find the index of the maximum logit (selection) for each observation
    selection = df.iloc[:, :num_classes].apply(np.argmax, axis=1)
    # Gather the logits for the selected classes by using .values and indexing directly
    selected_logits = df.values[np.arange(len(df)), selection]
    # Compute softmax probabilities for the selected class
    exp_values = np.exp(selected_logits)
    exp_sums = np.exp(df.iloc[:, :num_classes]).sum(axis=1)
    # Softmax probability for the selected class
    phat = exp_values / exp_sums
    return phat, selection

# Helper function to compute phat and selection for the lenet model
def process_lenet(df, num_classes):
    """
    Compute phat and selection for lenet models.
    Args:
    - df: DataFrame with probability values for each class.
    - num_classes: Number of classes.
    Returns:
    - df with phat and selection columns.
    """
    df['phat'] = df.iloc[:, :num_classes].max(axis=1)
    df['selection'] = df.iloc[:, :num_classes].apply(np.argmax, axis=1)
    return df

# Helper function to compute phat and selection for dla and res models
def process_dla_res(dla_df, res_df, num_classes):
    """
    Compute phat and selection for dla and resnet models.
    Args:
    - dla_df: DataFrame with dla logits (pre-softmax).
    - res_df: DataFrame with resnet logits (pre-softmax).
    - num_classes: Number of classes.
    Returns:
    - DataFrame with combined phat, selection_dla, and selection_res.
    """
    # Compute softmax probabilities and selections for dla and res models
    dla_phat, selection_dla = softmax_selected(dla_df, num_classes)
    res_phat, selection_res = softmax_selected(res_df, num_classes)
    # Average the softmax probabilities between dla and res models
    combined_phat = (dla_phat + res_phat) / 2
    # Create a DataFrame with phat, selection_dla, and selection_res
    combined_df = pd.DataFrame()
    combined_df['phat'] = combined_phat
    combined_df['selection_dla'] = selection_dla
    combined_df['selection_res'] = selection_res
    return combined_df

def process_actuals(actual_filename, dataset_name):
    """
    Process actuals file based on the dataset structure.
    For svhn and cifar, grab the 'label' column.
    For other datasets, grab the first column (no headers).
    """
    if dataset_name in ['svhn', 'cifar']:
        # Load actuals for svhn and cifar (with headers, 'label' column)
        actuals_df = pd.read_csv(actual_filename)
        return actuals_df['label'].round().astype(int)
    else:
        # Load actuals for other datasets (no headers, use first column)
        actuals_df = pd.read_csv(actual_filename, header=None)
        return actuals_df.iloc[:, 0].round().astype(int)

# Main function to process dataset
def process_dataset(dataset_name, pullsource="test", num_classes=10):
    """
    Process the dataset and extract the best and worst examples.
    Args:
    - dataset_name: Name of the dataset (e.g., "mnist", "kmnist", "fashionmnist", "cifar", "svhn")
    - pullsource: The source of the dataset ("test" by default)
    - num_classes: Number of classes in the dataset (default is 10)
    Returns:
    - best: Simplified DataFrame with the best two examples
    - worst: Simplified DataFrame with the worst two examples
    """
    # Determine the ds_rate and filenames based on the dataset_name
    if dataset_name in ["mnist", "kmnist", "fashionmnist"]:
        ds_rate = 7
        prob_filename = f"../tmp/probs/{dataset_name}_{pullsource}{ds_rate}_probs.csv"
        # Load the lenet model probabilities
        probs_df = pd.read_csv(prob_filename, header=None)
        # Process lenet data
        processed_df = process_lenet(probs_df, num_classes)
        # Add the 'selection' column and call it 'selection' for mnist-style
        selection_column = 'selection'
    elif dataset_name in ["cifar", "svhn"]:
        ds_rate = 8
        # Load dla and res probabilities
        dla_filename = f"../tmp/probs/{dataset_name}_{pullsource}{ds_rate}_dlaprobs.csv"
        res_filename = f"../tmp/probs/{dataset_name}_{pullsource}{ds_rate}_resprobs.csv"
        dla_df = pd.read_csv(dla_filename, header=None, dtype=float)
        res_df = pd.read_csv(res_filename, header=None, dtype=float)
        # Process dla and resnet data
        processed_df = process_dla_res(dla_df, res_df, num_classes)
        # For CIFAR and SVHN, we have two selection columns: 'selection_dla' and 'selection_res'
        selection_column = ['selection_dla', 'selection_res']
    # Add the 'obs' and 'dataset' columns
    n_rows = len(processed_df)
    processed_df['obs'] = np.arange(1, n_rows + 1)
    processed_df['dataset'] = dataset_name
    # Load the actual labels and add to the DataFrame
    actual_filename = f"../data/{dataset_name}/{dataset_name}_{pullsource}.csv"
    actuals_df = process_actuals(actual_filename, dataset_name)
    processed_df['actual'] = actuals_df
    # Sort the DataFrame by dataset and phat
    processed_df = processed_df.sort_values(by=['dataset', 'phat'])
    # Split the data based on correct classifications
    if dataset_name in ["cifar", "svhn"]:
        # For CIFAR/SVHN, both DLA and ResNet selections need to be correct
        splits_correct = processed_df[
            (processed_df['actual'] == processed_df['selection_dla']) &
            (processed_df['actual'] == processed_df['selection_res'])
        ]
    else:
        # For MNIST-style datasets, use the single 'selection' column
        splits_correct = processed_df[processed_df['actual'] == processed_df[selection_column]]
    # Get the best and worst examples
    worst = processed_df.head(2)[['obs', 'phat', 'selection', 'actual'] if isinstance(selection_column, str) else ['obs', 'phat', 'selection_dla', 'selection_res', 'actual']]
    best = splits_correct.tail(2)[['obs', 'phat', 'selection', 'actual'] if isinstance(selection_column, str) else ['obs', 'phat', 'selection_dla', 'selection_res', 'actual']]
    return best, worst

# Implement for different small image datasets
# Depending on your probabilities, you may get different "best" and "worst" cases
# than in the paper. For the purposes of replicating the published study
# the best and worst cases are hard-coded in the print_bestworst scripts.

for dataset in ["mnist", "kmnist", "fashionmnist", "cifar", "svhn"]:
    best, worst = process_dataset(dataset)
    print(f"Best examples for {dataset}:")
    print(best)
    print(f"Worst examples for {dataset}:")
    print(worst)

