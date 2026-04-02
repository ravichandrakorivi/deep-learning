# Import built-in modules from Python's standard library
import gzip
import pickle
import random

# Import Third-party libraries
import numpy as np

np.random.seed(42)
random.seed(42)

def update_mini_batch(a, b):
    ...

def evaluate(a):
    ...

def vectorized_label(label):
    label_vector = np.zeros((10, 1))
    label_vector[label] = 1.0
    return label_vector

with gzip.open('../data/mnist.pkl.gz', 'rb') as f:
    training_data, validation_data, testing_data = pickle.load(f, encoding='latin1')

training_features = [np.reshape(feature, (784, 1)) for feature in training_data[0]]
training_labels = [vectorized_label(label) for label in training_data[1]]
validation_features = [np.reshape(features, (784, 1)) for features in validation_data[0]]
testing_features = [np.reshape(features, (784, 1)) for features in testing_data[0]]

train_data = list(zip(training_features, training_labels))
val_data = list(zip(validation_features, validation_data[1]))
test_data = list(zip(testing_features, testing_data[1]))

sizes = [784, 30, 10]
biases = [np.random.randn(size, 1) for size in sizes[1:]]
weights = [np.random.randn(next_layer, curr_layer) for curr_layer, next_layer in zip(sizes[:-1], sizes[1:])]

# weights = [np.random.randn(next_layer, curr_layer) / np.sqrt(curr_layer) for curr_layer, next_layer in zip(sizes[:-1], sizes[1:])]

epochs = 30
mini_batch_size = 10
eta = 3.0

if testing_data: n_test = len(testing_data)
n = len(training_data)
for j in range(epochs):
    random.shuffle(training_data)
    mini_batches = [
        training_data[k:k+mini_batch_size]
        for k in range(0, n, mini_batch_size)]
    for mini_batch in mini_batches:
        update_mini_batch(mini_batch, eta)
    if testing_data:
        print(f"Epoch {j}: {evaluate(testing_data)} / {n_test}")
    else:
        print("Epoch {j} complete")


