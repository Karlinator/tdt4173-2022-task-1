import math
from typing import Callable
import numpy as np 
import numpy.typing as npt
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:
    theta = npt.NDArray
    iterations: int
    alpha: float | None
    
    def __init__(self, iterations: int = 1000, alpha: float | None = None):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.iterations = iterations
        self.alpha = alpha
        
    def fit(self, X: npt.NDArray, y: npt.NDArray):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """

        # self.theta = np.zeros((X.shape[1],))
        self.theta = np.random.random_sample(size=(X.shape[1]))

        alpha = self.alpha or 2 / np.linalg.norm(X)
        print(f'Using alpha={alpha:.3f}')

        for i in range(self.iterations):
            self.theta -= alpha * (X.T @ (self.predict(X) - y))
            


            if i % math.floor(self.iterations / 10) == 0:
                print(f"\n\nFinished iteration {i}")
                print(f'Parameters: {self.theta}')
                print(f'Accuracy: {binary_accuracy(y_true=y, y_pred=self.predict(X), threshold=0.5) :.3f}')
                print(f'Cross Entropy: {binary_cross_entropy(y_true=y, y_pred=self.predict(X)) :.3f}')
                # pass
        print(f"Finished iteration {self.iterations}")
        print(f"Final parameters: {self.theta}")
    
    def predict(self, X: npt.NDArray) -> npt.NDArray:
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
        X = (X - X.mean()) / (X.std())
        return sigmoid(X @ self.theta)
        

        
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
    

def binary_cross_entropy(y_true: npt.NDArray, y_pred: npt.NDArray, eps=1e-15):
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

        
