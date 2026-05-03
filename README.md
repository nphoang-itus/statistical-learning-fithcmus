# Statistical Learning Assignments (FITHCMUS)

This repository contains course assignments and exercises for the Statistical Learning course at the Faculty of Information Technology, Ho Chi Minh University of Science (FITHCMUS). The projects cover fundamental concepts in machine learning and deep learning, including practical implementations and experiments.

## Repository Structure

- **LA1-LogisticRegression/**  
  Implementation of logistic regression for binary classification.  
  - Includes configuration, training data, model parameters, and evaluation results.
  - Example metrics: Accuracy (0.83), Precision (0.78), Recall (0.91), F1-score (0.84).

- **LA2-MNIST/**  
  Handwritten digit classification on the MNIST dataset using TensorFlow/Keras.  
  - Implements both Logistic Regression and a simple Neural Network (two hidden layers).
  - Contains code, trained models, and result summaries.
  - **Note:** The MNIST data file (`data/mnist.npz`) is not included in this repository due to GitHub's file size limits. To run this lab, download the MNIST dataset file manually:
    1. Download `mnist.npz`: read [https://stackoverflow.com](https://stackoverflow.com/questions/40690203/how-can-i-import-the-mnist-dataset-that-has-been-manually-downloaded#:~:text=1%2018%2029-,Comments,another%20location%20to%20Your%20liking.)
    2. Place the file in the `LA2-MNIST/data/` directory.
  - Example results:  
    | Model    | Accuracy | Precision | Recall | F1-score |
    |----------|----------|-----------|--------|----------|
    | Logistic | 0.9127   | 0.9126    | 0.9127 | 0.9124   |
    | NN       | 0.9773   | 0.9775    | 0.9773 | 0.9773   |

- **SP1-LinearRegression/**  
  Linear regression assignment, including a Jupyter notebook for data analysis and model training.