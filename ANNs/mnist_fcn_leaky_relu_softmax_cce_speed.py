# Imports
import os
import gzip
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# RNG
rng = np.random.default_rng(42)

# -----------------------------
# Load MNIST (from files)
# -----------------------------
def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28, 28)

def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)

# Download
base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
files = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
]

os.makedirs("mnist_data", exist_ok=True)

for file in files:
    path = os.path.join("mnist_data", file)
    if not os.path.exists(path):
        urllib.request.urlretrieve(base_url + file, path)

# Load
X_train = load_images("mnist_data/train-images-idx3-ubyte.gz")
y_train = load_labels("mnist_data/train-labels-idx1-ubyte.gz")
X_test  = load_images("mnist_data/t10k-images-idx3-ubyte.gz")
y_test  = load_labels("mnist_data/t10k-labels-idx1-ubyte.gz")

# -----------------------------
# Preprocessing
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=10000, stratify=y_train, random_state=42
)

# Normalize
X_train = X_train / 255.0
X_val   = X_val / 255.0
X_test  = X_test / 255.0

# Flatten
X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
X_val   = X_val.reshape(X_val.shape[0], -1).astype(np.float32)
X_test  = X_test.reshape(X_test.shape[0], -1).astype(np.float32)

# One-hot
def one_hot(y):
    return np.eye(10)[y].T   # shape (10, N)

Y_train = one_hot(y_train)
Y_val   = one_hot(y_val)
Y_test  = one_hot(y_test)

# Transpose inputs → (features, samples)
X_train = X_train.T
X_val   = X_val.T
X_test  = X_test.T

# -----------------------------
# Model parameters
# -----------------------------
sizes = [784, 30, 10]

# He init
W1 = rng.normal(0, np.sqrt(2 / sizes[0]), size=(sizes[1], sizes[0]))
b1 = np.zeros((sizes[1], 1))

W2 = rng.normal(0, np.sqrt(2 / sizes[1]), size=(sizes[2], sizes[1]))
b2 = np.zeros((sizes[2], 1))

# -----------------------------
# Activations
# -----------------------------
def leaky_relu(z):
    return np.where(z > 0, z, 0.01*z)

def leaky_relu_prime(z):
    return np.where(z > 0, 1.0, 0.01)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

# -----------------------------
# Forward
# -----------------------------
def forward(X):
    Z1 = W1 @ X + b1
    A1 = leaky_relu(Z1)

    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

# -----------------------------
# Loss
# -----------------------------
def compute_loss(A2, Y):
    return -np.mean(np.sum(Y * np.log(A2 + 1e-9), axis=0))

# -----------------------------
# Backward
# -----------------------------
def backward(X, Y, Z1, A1, A2):
    m = X.shape[1]

    dZ2 = A2 - Y
    dW2 = (dZ2 @ A1.T) / m
    db2 = np.mean(dZ2, axis=1, keepdims=True)

    dZ1 = (W2.T @ dZ2) * leaky_relu_prime(Z1)
    dW1 = (dZ1 @ X.T) / m
    db1 = np.mean(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

# -----------------------------
# Accuracy
# -----------------------------
def accuracy(X, Y):
    _, _, _, A2 = forward(X)
    preds = np.argmax(A2, axis=0)
    true  = np.argmax(Y, axis=0)
    return np.mean(preds == true)

# -----------------------------
# Training
# -----------------------------
epochs = 30
batch_size = 64
eta = 0.1

n = X_train.shape[1]

for epoch in range(epochs):
    # Shuffle
    perm = rng.permutation(n)
    X_train = X_train[:, perm]
    Y_train = Y_train[:, perm]

    for i in range(0, n, batch_size):
        X_batch = X_train[:, i:i+batch_size]
        Y_batch = Y_train[:, i:i+batch_size]

        Z1, A1, Z2, A2 = forward(X_batch)
        dW1, db1, dW2, db2 = backward(X_batch, Y_batch, Z1, A1, A2)

        # Update
        W1 -= eta * dW1
        b1 -= eta * db1
        W2 -= eta * dW2
        b2 -= eta * db2

    val_acc = accuracy(X_val, Y_val)
    val_loss = compute_loss(forward(X_val)[-1], Y_val)

    print(f"Epoch {epoch+1}: Acc={val_acc:.4f}, Loss={val_loss:.4f}")

# -----------------------------
# Test
# -----------------------------
test_acc = accuracy(X_test, Y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")