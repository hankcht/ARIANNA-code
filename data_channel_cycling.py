"""
Data Channel Cycling Module

This module provides functionality to augment training data by creating rotations
of the channel dimension (typically the 4 antenna channels in ARIANNA data).

For each event in the dataset, this creates multiple versions with different
channel orderings, effectively multiplying the training set size.
"""

import numpy as np


def cycle_channels(data, channel_axis=1):
    """
    Create augmented dataset by cycling through different channel permutations.
    
    For each event, creates 6 additional versions with different channel orderings:
    - [2, 1, 0, 3]
    - [0, 3, 2, 1]
    - [2, 3, 0, 1]
    - [3, 0, 1, 2]
    - [2, 3, 0, 1] (duplicate in specification, kept as is)
    - [1, 2, 3, 0]
    
    Plus the original [0, 1, 2, 3] ordering.
    
    Args:
        data (np.ndarray): Input data with shape [n_events, 256, 4] or [n_events, 4, 256]
        channel_axis (int): The axis index where channels are located (1 for [n, 4, 256], 
                           2 for [n, 256, 4]). Default is 1.
    
    Returns:
        np.ndarray: Augmented dataset with shape [n_events * 7, ...] where each original
                   event is followed by its 6 permutations.
    
    Examples:
        >>> data = np.random.randn(100, 4, 256)  # 100 events, 4 channels, 256 samples
        >>> augmented = cycle_channels(data, channel_axis=1)
        >>> augmented.shape
        (700, 4, 256)
        
        >>> data = np.random.randn(100, 256, 4)  # 100 events, 256 samples, 4 channels
        >>> augmented = cycle_channels(data, channel_axis=2)
        >>> augmented.shape
        (700, 256, 4)
    """
    
    # Define the channel permutations
    permutations = [
        [0, 1, 2, 3],  # Original
        [2, 1, 0, 3],
        [0, 3, 2, 1],
        [2, 3, 0, 1],
        [3, 0, 1, 2],
        [2, 3, 0, 1],  # Note: This is a duplicate of permutation index 3
        [1, 2, 3, 0]
    ]
    
    n_events = data.shape[0]
    n_permutations = len(permutations)
    
    # Validate channel axis
    if channel_axis not in [1, 2]:
        raise ValueError(f"channel_axis must be 1 or 2, got {channel_axis}")
    
    # Validate that the channel dimension is 4
    if data.shape[channel_axis] != 4:
        raise ValueError(f"Expected 4 channels along axis {channel_axis}, "
                        f"but got {data.shape[channel_axis]}")
    
    print(f"Original data shape: {data.shape}")
    print(f"Creating {n_permutations} permutations per event (including original)")
    
    # Create list to hold all permuted versions
    augmented_data = []
    
    # For each event, create all permutations
    for i in range(n_events):
        event = data[i]
        
        for perm in permutations:
            if channel_axis == 1:
                # Shape is [4, 256], select along axis 0
                permuted_event = event[perm, :]
            else:  # channel_axis == 2
                # Shape is [256, 4], select along axis 1
                permuted_event = event[:, perm]
            
            augmented_data.append(permuted_event)
    
    # Stack all augmented events
    augmented_data = np.stack(augmented_data, axis=0)
    
    print(f"Augmented data shape: {augmented_data.shape}")
    print(f"Data multiplied by factor of {n_permutations}")
    
    return augmented_data


def cycle_channels_with_labels(data, labels, channel_axis=1):
    """
    Create augmented dataset by cycling channels, and replicate labels accordingly.
    
    This is a convenience function that augments both data and labels together,
    ensuring they remain synchronized.
    
    Args:
        data (np.ndarray): Input data with shape [n_events, 256, 4] or [n_events, 4, 256]
        labels (np.ndarray): Labels with shape [n_events] or [n_events, 1]
        channel_axis (int): The axis index where channels are located. Default is 1.
    
    Returns:
        tuple: (augmented_data, augmented_labels) both with n_events * 7 entries
    
    Examples:
        >>> data = np.random.randn(100, 4, 256)
        >>> labels = np.ones((100, 1))
        >>> aug_data, aug_labels = cycle_channels_with_labels(data, labels)
        >>> aug_data.shape, aug_labels.shape
        ((700, 4, 256), (700, 1))
    """
    
    n_permutations = 7  # Original + 6 permutations
    
    # Augment the data
    augmented_data = cycle_channels(data, channel_axis=channel_axis)
    
    # Replicate labels for each permutation
    if labels.ndim == 1:
        augmented_labels = np.repeat(labels, n_permutations)
    else:
        augmented_labels = np.repeat(labels, n_permutations, axis=0)
    
    print(f"Augmented labels shape: {augmented_labels.shape}")
    
    return augmented_data, augmented_labels


if __name__ == "__main__":
    # Test the function
    print("=" * 60)
    print("Testing cycle_channels function")
    print("=" * 60)
    
    # Test with shape [n_events, 4, 256]
    print("\nTest 1: Data shape [10, 4, 256]")
    test_data_1 = np.random.randn(10, 4, 256)
    augmented_1 = cycle_channels(test_data_1, channel_axis=1)
    print(f"Expected shape: (70, 4, 256), Got: {augmented_1.shape}")
    assert augmented_1.shape == (70, 4, 256), "Shape mismatch!"
    
    # Test with shape [n_events, 256, 4]
    print("\nTest 2: Data shape [10, 256, 4]")
    test_data_2 = np.random.randn(10, 256, 4)
    augmented_2 = cycle_channels(test_data_2, channel_axis=2)
    print(f"Expected shape: (70, 256, 4), Got: {augmented_2.shape}")
    assert augmented_2.shape == (70, 256, 4), "Shape mismatch!"
    
    # Test with labels
    print("\nTest 3: With labels")
    test_labels = np.ones((10, 1))
    augmented_data, augmented_labels = cycle_channels_with_labels(test_data_1, test_labels, channel_axis=1)
    print(f"Data shape: {augmented_data.shape}, Labels shape: {augmented_labels.shape}")
    assert augmented_data.shape[0] == augmented_labels.shape[0], "Data and labels size mismatch!"
    
    # Verify that permutations are applied correctly
    print("\nTest 4: Verify permutation correctness")
    simple_data = np.arange(4 * 3).reshape(1, 4, 3)  # 1 event, 4 channels, 3 samples
    simple_data[0] = [[0, 1, 2], [10, 11, 12], [20, 21, 22], [30, 31, 32]]
    print(f"Original event:\n{simple_data[0]}")
    
    augmented_simple = cycle_channels(simple_data, channel_axis=1)
    print(f"\nFirst permutation [0,1,2,3] (original):\n{augmented_simple[0]}")
    print(f"Second permutation [2,1,0,3]:\n{augmented_simple[1]}")
    print(f"Third permutation [0,3,2,1]:\n{augmented_simple[2]}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
