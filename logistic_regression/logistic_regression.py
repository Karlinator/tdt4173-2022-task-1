import numpy as np 
import numpy.typing as npt
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:
    theta: npt.ArrayLike
    iterations: int
    
    def __init__(self, n_parameters: int = 10, iterations: int = 10):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.theta = [0] * n_parameters # Just so I don't need an extra attribute for number of params I guess.
        self.iterations = iterations
        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """
        # Randomly initialize the parameters
        self.theta = np.random.random_sample(size=len(self.theta))

        for i in range(self.iterations):
            # Predict
            # Compute loss
            # Adjust

            if i % 10 == 0:
                print(f"Finished iteration {i}")
    
    def predict(self, X: pd.DataFrame):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        return sigmoid(self.theta * X)
        

        
# --- Some utility functions 

def binary_accuracy(y_true: npt.ArrayLike, y_pred: npt.ArrayLike, threshold=0.5):
    """
    Computes binary classification accuracy
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    

def binary_cross_entropy(y_true: npt.ArrayLike, y_pred: npt.ArrayLike, eps=1e-15):
    """
    Computes binary cross entropy 
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x: npt.ArrayLike | float):
    """
    Applies the logistic function element-wise
    
    Hint: highly related to cross-entropy loss 
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))

        
