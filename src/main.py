import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from math import log2
from pathlib import Path


def get_data() -> pd.DataFrame:
    """Get the PS4-1 dataset. It has three columns: x1 and x2 for features and y for response."""
    csv_path = Path(Path(__file__).parents[1], 'data', 'PS4_1_Train.csv')
    return pd.read_csv(csv_path)


def find_best_split(df: pd.DataFrame, verbose=False) -> None:
    """Print the best splitting option for either response variable."""

    initial_entropy = entropy(df['y'])
    print(f'Initial entropy with no splits == {initial_entropy}.')
    num_samples = df.shape[0]
    num_values = 20

    # Use these to track the best split found while iterating through both features.
    best_split_feature = None
    best_split_index = None
    best_split_value = None
    best_split_information_gain = None

    # Test each feature to find the best split
    for feature in ['x1', 'x2']:
        df_sorted = df.sort_values(by=feature, ignore_index=True)
        y_sorted = df_sorted['y']

        # Try 20 different splits for this feature
        for i in range(num_values):

            # Calculate the index and the feature's value on this splitting check
            split_index = round((i / (num_values - 1)) * (num_samples - 1))  # evenly spaced from 0 to 135
            split_value = df_sorted.at[split_index, feature]
            if verbose:
                print(f'Splitting feature {feature} at index {split_index} where {feature} == {split_value}:')

            # Actually split the dataset
            left_split = y_sorted[:split_index]
            right_split = y_sorted[split_index:]

            # Calculate the information gain from this split
            left_entropy = 0 if left_split.size == 0 else entropy(left_split)
            right_entropy = 0 if right_split.size == 0 else entropy(right_split)
            weighted_entropy = ((left_split.size * left_entropy) + (right_split.size * right_entropy)) / num_samples
            information_gain = initial_entropy - weighted_entropy

            # Optionally print out this split's results
            if verbose:
                print(f'Left branch: samples == {left_split.size}, entropy == {left_entropy}')
                print(f'Right branch: samples == {right_split.size}, entropy == {right_entropy}')
                print(f'weighted entropy == {weighted_entropy}, IG == {information_gain}')

            # Store these split results if it's the best one yet
            if best_split_information_gain is None or information_gain > best_split_information_gain:
                best_split_feature = feature
                best_split_index = split_index
                best_split_value = split_value
                best_split_information_gain = information_gain

    # Output the best split found in either feature
    print(f'The best option is to create a split at {best_split_feature} < {best_split_value}.')
    print(f'This occurs at index {best_split_index} when {best_split_feature} is sorted in ascending order.')
    print(f'Information gain == {best_split_information_gain}\n')


def entropy(y: pd.Series) -> float:
    """Find the entropy of a distribution of 0 and 1 values."""
    num_zeros = len(y[lambda x: x == 0])
    num_ones = len(y[lambda x: x == 1])
    p0 = num_zeros / (num_zeros + num_ones)
    p1 = num_ones / (num_zeros + num_ones)
    entropy_sum = 0
    entropy_sum -= 0 if p0 == 0 else p0 * log2(p0)
    entropy_sum -= 0 if p1 == 0 else p1 * log2(p1)
    return entropy_sum


def do_assignment() -> None:
    df = get_data()

    # Do problem 1
    print('Problem 1:')
    find_best_split(df, verbose=False)

    # Do problem 2
    print('Problem 2:')
    df2 = df.copy()
    samples = df2.sample(10, random_state=42)  # Use a seed for reproducibility
    for i in samples.index:
        df2.at[i, 'y'] = 1 if df.at[i, 'y'] == 0 else 0  # Flip the response in 10 random samples
    find_best_split(df2, verbose=False)

    # Make some figures to help with the analysis in problem 3.
    sns.set_theme()
    save_dir = Path(Path(__file__).parents[1], 'figures')

    ax = sns.scatterplot(df, x='x1', y='x2', hue='y')
    ax.set_title('Original dataset')
    ax.set(xlabel=r'$x_1$', ylabel=r'$x_2$')
    plt.savefig(Path(save_dir, 'original_dataset.png'))

    ax = sns.scatterplot(df2, x='x1', y='x2', hue='y')
    ax.set_title('Modified_dataset')
    ax.set(xlabel=r'$x_1$', ylabel=r'$x_2$')
    plt.savefig(Path(save_dir, 'modified_dataset.png'))


if __name__ == '__main__':
    do_assignment()
