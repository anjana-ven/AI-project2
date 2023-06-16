import decimal
import re
import operator
import math
import time
import os
import random

def euclidean_distance(training_instance, test_instance, current_features):
    """
    Calculates the Euclidean distance between a training instance and a test instance using the specified features.

    Args:
    - training_instance (list): Training instance.
    - test_instance (list): Test instance.
    - current_features (list): List of feature indices to consider.

    Returns:
    - float: Euclidean distance.
    """
    distance = 0
    for i in current_features:
        distance += math.pow(training_instance[i] - test_instance[i], 2)
    return math.sqrt(distance)

def nearest_neighbor_classification(train_data, test_instance, current_features):
    """
    Performs nearest neighbor classification by finding the nearest neighbor to the test instance among the training instances using the specified features.

    Args:
    - train_data (list): Training instances.
    - test_instance (list): Test instance.
    - current_features (list): List of feature indices to consider.

    Returns:
    - int: Predicted class label for the test instance.
    """
    distances = []
    for train_instance in train_data:
        distance = euclidean_distance(train_instance, test_instance, current_features)
        distances.append([distance, train_instance[0]])
    distances.sort(key=operator.itemgetter(0))
    return distances[0][1]

def k_fold_cross_validation(dataset, current_features, feature, choice):
    """
    Performs k-fold cross-validation using the specified feature subset.

    Args:
    - dataset (list): Dataset.
    - current_features (list): Current feature subset.
    - feature (int): Feature index to add or remove.
    - choice (int): 1 for feature addition, 2 for feature removal.

    Returns:
    - float: Accuracy of the classification.
    """
    correct_predictions = 0
    features = current_features[:]
    if choice == 1:
        features.append(feature)
    else:
        features.remove(feature)

    for k in range(len(dataset)):
        training_set = dataset[:]
        test_instance = training_set.pop(k)
        predicted_label = nearest_neighbor_classification(training_set, test_instance, features)
        if test_instance[0] == predicted_label:
            correct_predictions += 1
    accuracy = correct_predictions / float(len(dataset))
    return accuracy

def forward_selection(dataset, sample_size):
    """
    Performs feature selection using the forward selection algorithm.

    Args:
    - dataset (list): Dataset.
    - sample_size (int): Size of the dataset sample to consider (optional).

    Returns:
    None
    """
    best_features = []
    best_accuracy = 0
    current_features = []
    accuracies = []
    print("Beginning Forward Selection")

    if sample_size is not None:
        feature_size = sample_size
        sampled_dataset = random.sample(dataset, 1000)
    else:
        feature_size = len(dataset[0])
        sampled_dataset = dataset

    for i in range(1, feature_size):
        feature_to_add_this_level = -1
        highest_accuracy = 0

        for j in range(1, feature_size):
            if j not in current_features:
                accuracy = k_fold_cross_validation(sampled_dataset, current_features, j, 1)
                print(f"     Using feature(s) [{j}] accuracy is {accuracy}")
                if accuracy > highest_accuracy:
                    highest_accuracy = accuracy
                    feature_to_add_this_level = j

        if feature_to_add_this_level != -1:
            current_features.append(feature_to_add_this_level)
            print(f"On level {i}, feature {feature_to_add_this_level} was added to the current set")
            print(f"Feature set {current_features} was the best with accuracy: {highest_accuracy}")
            accuracies.append(highest_accuracy)
        if highest_accuracy > best_accuracy:
            best_features = current_features[:]
            best_accuracy = highest_accuracy

    print("Finished Forward Selection!!!")
    print("Best feature subset is:")
    print(best_features)
    print("Best accuracy is:")
    print(best_accuracy)

def backward_elimination(dataset, sample_size):
    """
    Performs feature selection using the backward elimination algorithm.

    Args:
    - dataset (list): Dataset.
    - sample_size (int): Size of the dataset sample to consider (optional).

    Returns:
    None
    """
    best_features = []
    best_accuracy = 0
    current_features = list(range(1, len(dataset[0])))
    accuracies = []
    print("Beginning Backward Elimination")

    if sample_size is not None:
        feature_size = sample_size
        sampled_dataset = random.sample(dataset, 2000)
    else:
        feature_size = len(dataset[0])
        sampled_dataset = dataset

    for i in range(1, feature_size):
        feature_to_remove_this_level = -1
        highest_accuracy = 0

        for j in range(1, feature_size):
            if j in current_features:
                accuracy = k_fold_cross_validation(sampled_dataset, current_features, j, 2)
                print(f"     Using feature(s) [{j}] accuracy is {accuracy}")
                if accuracy > highest_accuracy:
                    highest_accuracy = accuracy
                    feature_to_remove_this_level = j

        if feature_to_remove_this_level != -1:
            current_features.remove(feature_to_remove_this_level)
            print(f"On level {i}, feature {feature_to_remove_this_level} was removed from the current set with accuracy: {highest_accuracy}")
            accuracies.append(highest_accuracy)
        if highest_accuracy > best_accuracy:
            best_features = current_features[:]
            best_accuracy = highest_accuracy

    print("Finished Backward Elimination!!!")
    print("Best feature subset is:")
    print(best_features)
    print("Best accuracy is:")
    print(best_accuracy)

def normalize_dataset(dataset):
    """
    Normalizes the dataset by subtracting the mean and dividing by the standard deviation.

    Args:
    - dataset (list): Dataset to be normalized.

    Returns:
    - list: Normalized dataset.
    """
    normalized_dataset = dataset
    averages = [0.00] * (len(normalized_dataset[0]) - 1)
    stds = [0.00] * (len(normalized_dataset[0]) - 1)
    for instance in normalized_dataset:
        for j in range(1, len(instance)):
            averages[j - 1] += instance[j]
    for i in range(len(averages)):
        averages[i] = averages[i] / len(normalized_dataset)
    for instance in normalized_dataset:
        for j in range(1, len(instance)):
            stds[j - 1] += pow((instance[j] - averages[j - 1]), 2)
    for i in range(len(stds)):
        stds[i] = math.sqrt(stds[i] / len(normalized_dataset))
    for instance in normalized_dataset:
        for j in range(1, len(instance)):
            instance[j] = (instance[j] - averages[j - 1]) / stds[j - 1]
    return normalized_dataset

if __name__ == '__main__':
    file_path = '/content/sample_data/CS170_XXXlarge_Data__18.txt'
    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
    with open(file_path) as file:
        data = file.readlines()
    row = []
    dataset = []
    data = [x.strip() for x in data]
    for line in data:
        values = re.split(" +", line)
        for v in values:
            val = float(decimal.Decimal(v))
            row.append(val)
        dataset.append(row)
        row = []
    dataset = normalize_dataset(dataset)

    print("""Enter the algorithm:
    1) Forward Selection
    2) Backward Selection""")
    algorithm_choice = int(input("Enter your choice: "))
    print(f"This dataset has {len(dataset[0]) - 1} features (not including class attributes), with {len(dataset)} instances.")
    print("Normalizing the data... Done!")
    start_time = time.time()

    sample_size = None
    if file_size > 1048576:  # 1 MB = 1048576 bytes
        sample_size = 20

    if algorithm_choice == 1:
        forward_selection(dataset, sample_size)
    elif algorithm_choice == 2:
        backward_elimination(dataset, sample_size)
    print("--- %.1f sec ---" % (time.time() - start_time))
