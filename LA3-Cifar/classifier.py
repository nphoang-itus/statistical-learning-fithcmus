from __future__ import annotations

import abc
from typing import Optional

import numpy as np
import tensorflow as tf


class MNISTClassifier(abc.ABC):
    """Base class for MNIST digit classifiers."""

    def __init__(self):
        self.model: Optional[tf.keras.Model] = None

    @abc.abstractmethod
    def build_model(self) -> tf.keras.Model:
        """Build and return a compiled Keras model."""

    def train(self, x_train: np.ndarray, y_train: np.ndarray,
              epochs: int = 10, batch_size: int = 128,
              validation_split: float = 0.1) -> tf.keras.callbacks.History:
        """Train the model on the given data.

        TODO: Implement this method.
        - If self.model is None, call self.build_model() to create it.
        - Use the model's fit() method with the provided parameters.
        - Return the History object from fit().
        """
        raise NotImplementedError

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate the model on the test data.

        TODO: Implement this method.
        - Raise RuntimeError if self.model is None.
        - Use the model's evaluate() method to get loss and accuracy.
        - Use the model's predict() method and np.argmax to get predicted labels.
        - Return a dict with keys: "loss", "accuracy", "y_pred".
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Save the model to the given file path.

        TODO: Implement this method.
        - Raise RuntimeError if self.model is None.
        - Use the model's save() method.
        """
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Load a model from the given file path.

        TODO: Implement this method.
        - Use tf.keras.models.load_model() and assign to self.model.
        """
        raise NotImplementedError


class LogisticRegressionClassifier(MNISTClassifier):
    """Logistic regression (single dense layer with softmax)."""

    def build_model(self) -> tf.keras.Model:
        """Build a logistic regression model for MNIST.

        TODO: Implement this method.
        - Create a Sequential model with:
          - Input layer accepting 784-dimensional vectors.
          - A single Dense output layer with 10 units and softmax activation.
        - Compile with optimizer="sgd", loss="sparse_categorical_crossentropy",
          and metrics=["accuracy"].
        - Return the compiled model.
        """
        raise NotImplementedError


class NeuralNetworkClassifier(MNISTClassifier):
    """Simple feedforward neural network."""

    def build_model(self) -> tf.keras.Model:
        """Build a simple neural network for MNIST.

        TODO: Implement this method.
        - Create a Sequential model with:
          - Input layer accepting 784-dimensional vectors.
          - Dense hidden layer with 128 units and ReLU activation.
          - Dense hidden layer with 64 units and ReLU activation.
          - Dense output layer with 10 units and softmax activation.
        - Compile with optimizer="adam", loss="sparse_categorical_crossentropy",
          and metrics=["accuracy"].
        - Return the compiled model.
        """
        raise NotImplementedError


class CIFARMLPClassifier(MNISTClassifier):
    """Baseline MLP for CIFAR-10 classification."""

    def build_model(self) -> tf.keras.Model:
        """Build a baseline MLP model for CIFAR-10.

        TODO: Implement this method.
        - Create a Sequential model with:
          - Input layer accepting 32x32x3 images.
          - Flatten layer.
          - Dense hidden layer with 512 units and ReLU activation.
          - Dense hidden layer with 256 units and ReLU activation.
          - Dense output layer with 10 units and softmax activation.
        - Compile with optimizer="adam", loss="sparse_categorical_crossentropy",
          and metrics=["accuracy"].
        - Return the compiled model.
        """
        raise NotImplementedError


class CIFARCNNClassifier(MNISTClassifier):
    """Convolutional neural network for CIFAR-10 classification."""

    def build_model(self) -> tf.keras.Model:
        """Build a CNN model for CIFAR-10.

        TODO: Implement this method.
        - Create a Sequential model with the following blocks:
          - Input layer accepting 32x32x3 images.
          Block 1:
          - Conv2D(32, 3, padding="same", activation="relu") + BatchNormalization
          - Conv2D(32, 3, padding="same", activation="relu") + BatchNormalization
          - MaxPooling2D(2) + Dropout(0.2)
          Block 2:
          - Conv2D(64, 3, padding="same", activation="relu") + BatchNormalization
          - Conv2D(64, 3, padding="same", activation="relu") + BatchNormalization
          - MaxPooling2D(2) + Dropout(0.3)
          Block 3:
          - Conv2D(128, 3, padding="same", activation="relu") + BatchNormalization
          - Conv2D(128, 3, padding="same", activation="relu") + BatchNormalization
          - MaxPooling2D(2) + Dropout(0.4)
          Head:
          - Flatten
          - Dense(256, activation="relu") + BatchNormalization + Dropout(0.5)
          - Dense(10, activation="softmax")
        - Compile with optimizer="adam", loss="sparse_categorical_crossentropy",
          and metrics=["accuracy"].
        - Return the compiled model.
        """
        raise NotImplementedError
