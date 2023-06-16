import decimal
import re
import operator
import math
import time
import os

def euclidean_distance(training_instance, test_instance, current_features):
    """
    Calculates the Euclidean distance between two instances.

    Args:
    - training_instance (list): Training instance.
    - test_instance (list): Test instance.
    - current_features (list): List of indices of features to consider.

    Returns:
    - float: Euclidean distance.
    """
    dist = 0
    for i in current_features:
        dist += math.pow(training_instance[i] - test_instance[i], 2)
    return math.sqrt(dist)

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
    

def forward_selection(dataset):
    """
    Performs feature selection using the forward selection algorithm.

    Args:
    - dataset (list): Dataset.

    Returns:
    None
    """
    best_features = []
    best_accuracy = 0
    current_featuresent_features = []
    feature_size = len(dataset[0])
    accuracies = []
    print("Beginning Forward Selection")
    for i in range(1, feature_size):
        feature_to_add_this_level = -1
        highest_accuracy = 0

        for j in range(1, feature_size):
            if j not in current_featuresent_features:
                accuracy = k_fold_cross_validation(dataset, current_featuresent_features, j, 1)
                print(f"     Using feature(s) [{j}] accuracy is {accuracy}")
                if accuracy > highest_accuracy:
                    highest_accuracy = accuracy
                    feature_to_add_this_level = j

        if feature_to_add_this_level != -1:
            current_featuresent_features.append(feature_to_add_this_level)
            print(f"On level {i}, added feature {feature_to_add_this_level} to current_featuresent set")
            print(f"Feature set {current_featuresent_features} was best with accuracy: {highest_accuracy}")
            accuracies.append(highest_accuracy)
        if highest_accuracy > best_accuracy:
            best_features = current_featuresent_features[:]
            best_accuracy = highest_accuracy
    print("Finished Forward Selection!!!")
    print("Best feature subset is: ")
    print(best_features)
    print("Best accuracy is: ")
    print(best_accuracy)

def backward_elimination(dataset):
    """
    Performs feature selection using the backward elimination algorithm.

    Args:
    - dataset (list): Dataset.

    Returns:
    None
    """
    best_features = []
    best_accuracy = 0
    current_featuresent_features = list(range(1, len(dataset[0])))
    feature_size = len(dataset[0])
    accuracies = []
    print("Beginning Backward Elimination")
    for i in range(1, feature_size):
        feature_to_remove_this_level = -1
        highest_accuracy = 0

        for j in range(1, feature_size):
            if j in current_featuresent_features:
                accuracy = k_fold_cross_validation(dataset, current_featuresent_features, j, 2)
                print(f"     Using feature(s) [{j}] accuracy is {accuracy}")
                if accuracy > highest_accuracy:
                    highest_accuracy = accuracy
                    feature_to_remove_this_level = j

        if feature_to_remove_this_level != -1:
            current_featuresent_features.remove(feature_to_remove_this_level)
            print(f"On level {i}, removed feature {feature_to_remove_this_level} from current_featuresent set with accuracy: {highest_accuracy}")
            accuracies.append(highest_accuracy)
        if highest_accuracy > best_accuracy:
            best_features = current_featuresent_features[:]
            best_accuracy = highest_accuracy

    print("Finished Backward Elimination!!!")
    print("Best feature subset is: ")
    print(best_features)
    print("Best accuracy is: ")
    print(best_accuracy)

def normalize_dataset(active_dataset):
    """
    Normalizes the dataset.

    Args:
    - active_dataset (list): Dataset.

    Returns:
    - list: Normalized dataset.
    """
    dataset = active_dataset
    averages = [0.00] * (len(dataset[0]) - 1)
    stds = [0.00] * (len(dataset[0]) - 1)
    for instance in dataset:
        for j in range(1, len(instance)):
            averages[j - 1] += instance[j]
    for i in range(len(averages)):
        averages[i] = averages[i] / len(dataset)
    for instance in dataset:
        for j in range(1, len(instance)):
            stds[j - 1] += pow((instance[j] - averages[j - 1]), 2)
    for i in range(len(stds)):
        stds[i] = math.sqrt(stds[i] / len(dataset))
    for instance in dataset:
        for j in range(1, len(instance)):
            instance[j] = (instance[j] - averages[j - 1]) / stds[j - 1]
    return dataset

if __name__ == '__main__':
    filename = input("Enter file name:")
    with open(filename) as f:
        data_lines = f.readlines()
    row = []
    dataset = []
    data_lines = [x.strip() for x in data_lines]
    for line in data_lines:
        values = re.split(" +", line)
        for v in values:
            val = float(decimal.Decimal(v))
            row.append(val)
        dataset.append(row)
        row = []
    dataset = normalize_dataset(dataset)

    print("""Enter the algorithm:
        1) Forward Selection
        2) Backward Elimination""")
    choice = int(input("Enter your choice: "))
    print(f"This dataset has {len(dataset[0]) - 1} features (not including class attributes), with {len(dataset)} instances.")
    print("Normalizing the data... Done!")
    start_time = time.time()
    if choice == 1:
        forward_selection(dataset)
    elif choice == 2:
        backward_elimination(dataset)
    print("--- %.1f sec ---" % (time.time() - start_time))
