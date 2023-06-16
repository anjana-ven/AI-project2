import operator
import math
import time
import os
import random
import pandas as pd


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


def forward_selection(table, sample_size):
    # Perform feature selection using the forward selection algorithm
    final_features = []
    final_accuracy = 0
    current_features = []
    accuracies = []
    print("Beginning Search")

    if sample_size is not None:
        feature_size = sample_size
        sampled_table = random.sample(table, 3000)
    else:
        feature_size = len(table[0])
        sampled_table = table

    for i in range(0, feature_size):
        feature_to_add_this_level = -1
        best_accuracy = 0

        for j in range(0, feature_size):
            if j not in current_features:
                #print("Table features currently using:", table[0][j])
                accuracy = k_fold_cross_validation(sampled_table, current_features, j, 1)
                print(f"     Using feature(s) [{j}] accuracy is {accuracy}")
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    feature_to_add_this_level = j

        if feature_to_add_this_level != -1:
            current_features.append(feature_to_add_this_level)
            print(f"On level {i} I added feature {feature_to_add_this_level} to current set")
            print(f"Feature set {current_features} was best with accuracy: {best_accuracy}")
            accuracies.append(best_accuracy)
        if best_accuracy > final_accuracy:
            final_features = current_features[:]
            final_accuracy = best_accuracy

    print("Finished Search!!!")
    print("Best feature subset is: ")
    print(final_features)
    print("Best accuracy is: ")
    print(final_accuracy)


def backward_elimination(table, sample_size):
    # Perform feature selection using the backward elimination algorithm
    final_features = []
    final_accuracy = 0
    current_features = list(range(0, len(table[0])))
    accuracies = []
    print("Beginning Search")

    if sample_size is not None:
        feature_size = sample_size
        sampled_table = random.sample(table, 3000)
    else:
        feature_size = len(table[0])
        sampled_table = table

    for i in range(0, feature_size):
        feature_to_remove_this_level = -1
        best_accuracy = 0

        for j in range(0, feature_size):
            if j in current_features:
                #print("Table features currently using:", table[0][j])
                accuracy = k_fold_cross_validation(sampled_table, current_features, j, 2)
                print(f"     Using feature(s) [{j}] accuracy is {accuracy}")
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    feature_to_remove_this_level = j

        if feature_to_remove_this_level != -1:
            current_features.remove(feature_to_remove_this_level)
            print(f"On level {i} I removed feature {feature_to_remove_this_level} from current set with accuracy: {best_accuracy}")
            accuracies.append(best_accuracy)
        if best_accuracy > final_accuracy:
            final_features = current_features[:]
            final_accuracy = best_accuracy

    print("Finished Search!!!")
    print("Best feature subset is: ")
    print(final_features)
    print("Best accuracy is: ")
    print(final_accuracy)


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
    file = '/content/sample_data/diabetes_prediction_dataset.csv'  # Update the file path to the actual location of the dataset file
    file_size = os.path.getsize(file) if os.path.exists(file) else 0
    data = pd.read_csv(file)

    # Perform any necessary preprocessing steps
    # For example, removing catagorial features

    columns_to_drop = ['gender', 'hypertension', 'heart_disease', 'smoking_history', 'diabetes']  # Replace with the actual column names you want to drop for me these were the catagorial values
    data = data.drop(columns_to_drop, axis=1)

    # Extract the features and labels from the dataset
    table = data.values.tolist()
    table = normalize_dataset(table)

    print("""Enter the algorithm:
    1) Forward Selection
    2) Backward Selection""")
    n = int(input("Enter your choice: "))
    print(f"This dataset has {data.shape[1] - 1} features (not including class attributes), with {data.shape[0]} instances.")
    print("Normalizing the data... Done!")
    start_time = time.time()

    sample_size = None
    if file_size > 1048576:  # 1 MB = 1048576 bytes
        sample_size = 4

    if n == 1:
        forward_selection(table, sample_size)
    elif n == 2:
        backward_elimination(table, sample_size)
    print("--- %.1f sec ---" % (time.time() - start_time))
