import numpy as np

def cos_sim(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the cosine similarity between a vector `x` and each row of matrix `y`.

    Parameters:
    x (np.ndarray): A 1-dimensional array representing the vector.
    y (np.ndarray): A 2-dimensional array where each row is a vector to compare with `x`.

    Returns:
    float: The cosine similarity between `x` and each row of `y`.
    """
    return y.dot(x) / (np.linalg.norm(x) * np.linalg.norm(y, axis=1))

def relu(x: np.ndarray) -> np.ndarray:
    """
    Apply the Rectified Linear Unit (ReLU) activation function to the input array.

    Parameters:
    x (np.ndarray): The input array.

    Returns:
    np.ndarray: The output array after applying the ReLU function.
    """
    return np.maximum(0, x)

def normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalize the input array by dividing each element by the sum of all elements.

    Parameters:
    x (np.ndarray): The input array.

    Returns:
    np.ndarray: The normalized array.
    """
    return x / np.sum(x)

def z_score_outliers(x: np.ndarray, thr=3):
    """
    Identify outliers in the input array using the Z-score method.

    Parameters:
    x (np.ndarray): The input array.
    thr (float): The threshold for the Z-score, default is 3.

    Returns:
    np.ndarray: A boolean array indicating whether each element is an outlier.
    """
    z_scores = (x - x.mean()) / x.std()
    return np.abs(z_scores) > thr