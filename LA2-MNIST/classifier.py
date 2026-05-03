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
        # Build the model if it has not been instantiated yet
        if self.model is None:
            self.model = self.build_model()
            
        # Train the model and return the training history
        history = self.model.fit(
            x=x_train, y=y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )
        return history

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate the model on the test data.

        TODO: Implement this method.
        - Raise RuntimeError if self.model is None.
        - Use the model's evaluate() method to get loss and accuracy.
        - Use the model's predict() method and np.argmax to get predicted labels.
        - Return a dict with keys: "loss", "accuracy", "y_pred".
        """
        # Check if the model exists before evaluating
        if self.model is None:
            raise RuntimeError("Model has not been built or loaded yet.")
            
        # Evaluate to get loss and accuracy
        eval_results = self.model.evaluate(x_test, y_test, verbose=0)
        if isinstance(eval_results, (list, tuple)):
            loss = eval_results[0]
            accuracy = eval_results[1]
        else:
            loss = eval_results
            accuracy = 0.0 # By Default if the model has not been compile accuracy metric

        # Get predictions and convert probabilities to class labels
        predictions = self.model.predict(x_test, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        
        # Return the results as a dictionary
        return {
            "loss": loss,
            "accuracy": accuracy,
            "y_pred": y_pred
        }

    def save(self, path: str) -> None:
        """Save the model to the given file path.

        TODO: Implement this method.
        - Raise RuntimeError if self.model is None.
        - Use the model's save() method.
        """
        # Check if there is a model to save
        if self.model is None:
            raise RuntimeError("No model to save.")
        
        self.model.save(path)

    def load(self, path: str) -> None:
        """Load a model from the given file path.

        TODO: Implement this method.
        - Use tf.keras.models.load_model() and assign to self.model.
        """
        self.model = tf.keras.models.load_model(path)


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
        # Initialize the Sequential model (a series of consecutive layers)
        model = tf.keras.Sequential([
            # Input layer: accepting 784-dimensional vectors
            tf.keras.layers.Input(shape=(784,)),
            
            # Output Layer: a single Dense layer with 10 units and softmax activation.
            tf.keras.layers.Dense(10, activation="softmax")
        ])
        
        # Compile the model with optimized configurations and evaluate
        model.compile(
            optimizer="sgd",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        return model


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
        # Initialize the Sequential model
        model = tf.keras.Sequential([
            # Input layer: accepting 784-dimensional vectors
            tf.keras.layers.Input(shape=(784,)),
            
            # Hidden layer 1: 128 units and ReLU activation.
            tf.keras.layers.Dense(128, activation="relu"),
            
            # Hidden layer 2: 64 units and ReLU activation.
            tf.keras.layers.Dense(64, activation="relu"),
            
            # Output layer: 10 units and ReLU activation.
            tf.keras.layers.Dense(10, activation="softmax")
        ])
        
        # Compile the model with optimize 'adam'
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        return model
